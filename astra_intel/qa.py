"""Question answering: retrieve -> gate -> answer -> cite.

1. BM25 retrieves the best page-bound chunks (deterministic, so repeated
   questions retrieve the same evidence).
2. A grounding gate decides whether the retrieved text can support the question
   at all. If not, the answer is "not in the document" — no LLM is asked to guess.
3. The answer is an LLM phrasing of the passages (temperature 0, cite-or-refuse
   prompt) or, with no API key, the best-matching sentences quoted verbatim.
4. Every answer carries its source passages with page numbers and highlights.
"""
from __future__ import annotations

import re
import time

from . import db, llm
from .extract import _CITATION_MARK
from .nlp import BM25, proper_terms, query_terms, split_sentences, stem, tokenize, tokenize_spans

TOP_K = 6
MAX_PASSAGES = 8
NOT_FOUND = "The document doesn't contain this information."

_FOLLOWUP = re.compile(r"\b(it|its|they|their|them|this|that|those|these|he|she|there|one|ones|also|more)\b", re.I)
_STAT = re.compile(r"\b(percent|percentage|percentages|statistic|statistics|proportion|rate of|fraction|share of)\b", re.I)

_SUMMARY_WORDS = {"summaris", "summariz", "summary", "summar", "overview", "point", "gist", "brief", "tldr", "recap",
                  "key", "main", "important", "major", "quick", "short", "highlight", "takeaway", "outline", "everything"}
_SUMMARY_STEMS = {stem(w) for w in _SUMMARY_WORDS} | {"summarise", "summarize", "summary"}

_index_cache: dict = {}


def invalidate() -> None:
    _index_cache.clear()


def _index(scope):
    key = "all" if scope in (None, "all") else int(scope)
    if key not in _index_cache:
        chunks = db.chunks_for(None if key == "all" else key)
        _index_cache[key] = (chunks, BM25([c["text"] for c in chunks]))
    return _index_cache[key]


def is_summary_request(question: str) -> bool:
    """True when the question asks for a summary and names no topic of its own."""
    if re.fullmatch(r"\W*tl;?\s?dr\W*", question.strip(), re.I):
        return True
    raw = query_terms(question)
    return bool(raw) and any(t in _SUMMARY_STEMS for t in raw) and all(t in _SUMMARY_STEMS for t in raw)


def retrieval_terms(question: str, history: list[dict]) -> list[str]:
    terms = [t for t in query_terms(question) if t not in _SUMMARY_STEMS]
    prev = [h["content"] for h in history if h["role"] == "user"]
    if prev and (len(terms) < 2 or (len(terms) < 4 and _FOLLOWUP.search(question))):
        for t in query_terms(prev[-1]):
            if t not in terms:
                terms.append(t)
    return terms


def _highlights(text: str, terms: set[str]) -> list[list[int]]:
    spans = [[s, e] for st, s, e in tokenize_spans(text) if st in terms]
    merged: list[list[int]] = []
    for s, e in spans:
        if merged and s - merged[-1][1] <= 1:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return merged


def _coverage(bm: BM25, terms: list[str], text: str) -> float:
    present = set(tokenize(text))
    total = sum(bm.idf_of(t) for t in terms)
    return (sum(bm.idf_of(t) for t in terms if t in present) / total) if total else 0.0


def retrieve(scope, question: str, history: list[dict] | None = None) -> dict:
    chunks, bm = _index(scope)
    terms = retrieval_terms(question, history or [])
    if not chunks or not terms:
        return {"passages": [], "terms": terms, "tier": "none", "stat_ok": True, "coverage": 0.0, "bm": bm}
    ranked = bm.rank(terms, TOP_K)
    if ranked and (scope in (None, "all")):
        have = {chunks[i]["doc_id"] for i, _ in ranked}
        floor = ranked[0][1] * 0.4
        for doc_id in sorted({c["doc_id"] for c in chunks} - have):
            best = max(((i, bm.score(terms, i)) for i, c in enumerate(chunks) if c["doc_id"] == doc_id),
                       key=lambda x: x[1], default=None)
            if best and best[1] >= floor and len(ranked) < MAX_PASSAGES:
                ranked.append(best)
    tset = set(terms)
    passages = []
    for n, (i, sc) in enumerate(ranked, start=1):
        c = chunks[i]
        passages.append({"id": f"P{n}", "chunk_id": c["id"], "doc_id": c["doc_id"], "doc_name": c["doc_name"],
                         "page": c["page"], "text": c["text"], "score": round(sc, 3),
                         "highlights": _highlights(c["text"], tset)})
    best_chunk = max((_coverage(bm, terms, p["text"]) for p in passages), default=0.0)
    best_sent = 0.0
    for p in passages[:4]:
        for s in split_sentences(p["text"]):
            best_sent = max(best_sent, _coverage(bm, terms, s))
    coverage = max(best_chunk, best_sent)
    unseen = [t for t in terms if t not in bm.idf]
    off_topic = (len(terms) >= 2 and len(unseen) / len(terms) >= 0.34) or any(t not in bm.idf for t in proper_terms(question))
    tier = "high" if (best_chunk >= 0.6 or best_sent >= 0.5) else ("low" if coverage >= 0.2 else "none")
    if off_topic:
        tier = "none"
    stat_ok = True
    if _STAT.search(question):
        stat_ok = any(
            re.search(r"\d", s) and re.search(r"%|percent", s, re.I) and _coverage(bm, terms, s) >= 0.4
            for p in passages[:4] for s in split_sentences(p["text"])
        )
    return {"passages": passages, "terms": terms, "tier": tier, "stat_ok": stat_ok,
            "coverage": round(coverage, 3), "bm": bm}


def _extractive_answer(question: str, r: dict) -> tuple[str, list[str]]:
    bm, terms = r["bm"], r["terms"]
    cands = []
    for p in r["passages"][:5]:
        for s in split_sentences(p["text"]):
            if not (30 <= len(s) <= 420) or sum(ch.isalpha() for ch in s) / len(s) < 0.7 or _CITATION_MARK.search(s):
                continue
            present = set(tokenize(s))
            sc = sum(bm.idf_of(t) for t in terms if t in present) / (len(s) ** 0.25)
            cands.append((sc, -p["score"], p["id"], s))
    cands.sort(key=lambda x: (-x[0], x[1]))
    if not cands or cands[0][0] <= 0:
        return NOT_FOUND, []
    top = cands[0][0]
    out, seen, used = [], set(), []
    for sc, _, pid, s in cands:
        key = re.sub(r"\W+", "", s.lower())[:80]
        if sc < top * 0.55 or key in seen:
            continue
        seen.add(key)
        out.append(f"• {s} [{pid}]")
        used.append(pid)
        if len(out) == 3:
            break
    return "\n".join(out), used


SYSTEM_PROMPT = (
    "You answer questions about uploaded documents for a defence-technology analyst.\n"
    "RULES:\n"
    "1. Use ONLY the numbered passages provided. Never use outside knowledge, even if you know the answer.\n"
    "2. After every claim add the passage tag(s) that support it, like [P2] or [P1][P3].\n"
    "3. If the passages do not contain the answer, reply with exactly: NOT_IN_DOCUMENT\n"
    "4. Do not invent numbers, names, dates or programmes. If something is only partly covered, say what is "
    "covered and what is not.\n"
    "5. Be concise: 2-6 sentences or a short bullet list."
)


def _llm_answer(question: str, r: dict, history: list[dict], mode: str) -> tuple[str, list[str]]:
    ctx = "\n\n".join(f"[{p['id']}] ({p['doc_name']}, page {p['page']})\n{p['text']}" for p in r["passages"])
    msgs = []
    for h in history[-6:]:
        msgs.append({"role": h["role"], "content": h["content"]})
    clean: list[dict] = []
    for m in msgs:
        if clean and clean[-1]["role"] == m["role"]:
            clean[-1]["content"] += "\n" + m["content"]
        else:
            clean.append(dict(m))
    while clean and clean[0]["role"] != "user":
        clean.pop(0)
    if clean and clean[-1]["role"] == "user":
        clean.pop()
    clean.append({"role": "user", "content": f"PASSAGES:\n{ctx}\n\nQUESTION: {question}"})
    text = llm.complete(SYSTEM_PROMPT, clean, mode=mode)
    if text.strip().upper().startswith("NOT_IN_DOCUMENT"):
        return NOT_FOUND, []
    used = []
    for tag in re.findall(r"\[(P\d+)\]", text):
        if tag not in used:
            used.append(tag)
    return text, used


def _summary_answer(scope, scope_key: str, question: str, chosen: str, t0: float) -> dict:
    docs = db.list_documents() if scope_key == "all" else [{"id": int(scope_key)}]
    parts = []
    for d in docs:
        full = db.get_document(d["id"])
        if full:
            parts.append(full["summary"] if len(docs) == 1 else f"{full['name']}\n{full['summary']}")
    text = "\n\n".join(parts) or NOT_FOUND
    ms = round((time.perf_counter() - t0) * 1000)
    tech = {"requested_mode": chosen, "answer_engine": "stored document summary", "model": "n/a (generated at upload)",
            "query_terms": [], "chunks_searched": 0, "passages_retrieved": 0, "top_scores": [],
            "gate": {"tier": "high", "coverage": 1.0, "statistic_check": True,
                     "rule": "summary request: answered from the summary created at upload, no search needed"},
            "llm_called": False, "timing_ms": {"retrieval": 0, "llm": None, "total": ms}, "temperature": None}
    meta = {"status": "grounded", "mode": "extractive", "sources": [], "coverage": 1.0, "note": None, "tech": tech}
    db.add_message(scope_key, "user", question)
    db.add_message(scope_key, "assistant", text, meta)
    return {"answer": text, **meta}


def answer(scope, question: str, mode: str | None = "auto") -> dict:
    t0 = time.perf_counter()
    chosen = llm.resolve(mode)
    scope_key = "all" if scope in (None, "all") else str(scope)
    if is_summary_request(question):
        return _summary_answer(scope, scope_key, question, chosen, t0)
    history = db.get_history(scope_key, limit=8)
    r = retrieve(scope, question, history)
    t_retr = time.perf_counter()
    used_mode, note = "extractive", None
    status, text, used = "grounded", "", []
    llm_ms, llm_called = None, False

    local = chosen == "local"
    if not r["passages"] or r["tier"] == "none" or (local and not r["stat_ok"]):
        status, text = "not_found", NOT_FOUND
    else:
        if not local:
            try:
                t1 = time.perf_counter()
                llm_called = True
                text, used = _llm_answer(question, r, history, chosen)
                llm_ms = round((time.perf_counter() - t1) * 1000)
                used_mode = chosen
            except llm.LLMError as e:
                note = f"{e} Showing an extractive answer instead."
                local = True
        if local:
            text, used = _extractive_answer(question, r)
            if text != NOT_FOUND and r["tier"] == "low":
                status = "low_confidence"
        if text == NOT_FOUND:
            status = "not_found"

    sources = []
    for p in r["passages"]:
        q = {k: v for k, v in p.items() if k != "chunk_id"}
        q["cited"] = p["id"] in used
        q["weak"] = status in ("not_found", "low_confidence")
        sources.append(q)
    if status == "grounded":
        sources.sort(key=lambda s: (not s["cited"], int(s["id"][1:])))

    chunks, bm = _index(scope)
    tech = {
        "requested_mode": chosen,
        "answer_engine": used_mode if used_mode != "extractive" else "extractive (local)",
        "model": llm.providers()[used_mode]["model"] if used_mode in ("anthropic", "openai") else "none (no LLM used)",
        "query_terms": r["terms"],
        "chunks_searched": len(chunks),
        "passages_retrieved": len(r["passages"]),
        "top_scores": [{"id": p["id"], "page": p["page"], "bm25": p["score"]} for p in r["passages"]],
        "gate": {"tier": r["tier"], "coverage": r["coverage"], "statistic_check": r["stat_ok"],
                 "rule": "high: best chunk >=0.60 or sentence >=0.50 term coverage; low: >=0.20; else none"},
        "llm_called": llm_called,
        "timing_ms": {"retrieval": round((t_retr - t0) * 1000),
                      "llm": llm_ms,
                      "total": round((time.perf_counter() - t0) * 1000)},
        "temperature": 0 if llm_called else None,
    }
    meta = {"status": status, "mode": used_mode, "sources": sources, "coverage": r["coverage"], "note": note, "tech": tech}
    db.add_message(scope_key, "user", question)
    db.add_message(scope_key, "assistant", text, meta)
    return {"answer": text, **meta}
