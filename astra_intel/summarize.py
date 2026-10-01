"""Document summary and key topics. LLM when available, extractive otherwise."""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

from . import llm
from .nlp import META_WORDS, STOPWORDS, _TOKEN_RE, split_sentences, stem

MAX_LLM_CHARS = 14000


def _term_stats(pages: list[str]):
    """tf and page-spread per stem, plus most common surface form."""
    tf: Counter = Counter()
    page_df: Counter = Counter()
    forms: dict[str, Counter] = defaultdict(Counter)
    for text in pages:
        seen = set()
        for m in _TOKEN_RE.finditer(text):
            raw = m.group()
            low = raw.lower()
            if low in STOPWORDS or low in META_WORDS or low.isdigit() or len(low) < 3:
                continue
            s = stem(low)
            tf[s] += 1
            forms[s][raw if raw.isupper() and len(raw) > 1 else low] += 1
            seen.add(s)
        page_df.update(seen)
    return tf, page_df, forms


_KW_STOP = {"retrieved", "original", "archived", "isbn", "doi", "issn", "edition", "press", "also", "new", "see", "ref",
            "external", "links", "reference", "references", "wikipedia", "wikimedia", "web", "org", "com", "pdf",
            "citation", "accessed", "journal", "university"}
_KW_STOP = {stem(w) for w in _KW_STOP}


def keywords(pages: list[str], n: int = 8) -> list[str]:
    tf, page_df, forms = _term_stats(pages)
    scored = {s: c * math.log(1 + page_df[s]) for s, c in tf.items() if c >= 3 and s not in _KW_STOP}
    top = sorted(scored, key=lambda s: (-scored[s], s))[:n]
    return [forms[s].most_common(1)[0][0] for s in top]


def _good_sentence(s: str) -> bool:
    if not (45 <= len(s) <= 380):
        return False
    if not s[0].isupper():
        return False
    letters = sum(ch.isalpha() for ch in s)
    if letters / len(s) < 0.75:  # tables, coordinates, citations
        return False
    if re.search(r"https?://|://|\(htt|\bISBN\b|\bdoi\b|\bRetrieved\b|\bArchived\b", s, re.I):
        return False
    if re.match(r"(See also|External links|References|Further reading)\b", s, re.I):
        return False
    words = s.split()
    caps = sum(1 for w in words[1:] if w[:1].isupper())
    if caps / max(1, len(words) - 1) > 0.3:  # image captions / name lists
        return False
    return s.rstrip().endswith((".", "!", "?"))


_DEF_VERB = re.compile(r"\b(is|are|refers to|was|were)\b")


def _trim_caption_prefix(s: str) -> str:
    """PDF text often glues a figure caption/title in front of the first sentence.
    Cut back to the latest Capitalised word that still leaves a subject of >=4 words before the verb."""
    m = _DEF_VERB.search(s)
    if not m or len(s) < 120:
        return s
    toks = [(t.start(), t.group()) for t in re.finditer(r"\S+", s[: m.start()])]
    best = 0
    for i, (pos, w) in enumerate(toks):
        if i and re.fullmatch(r"[A-Z][a-z]+", w) and len(toks) - i >= 4:
            best = pos
    return s[best:]


def extractive_summary(pages: list[str], n_body: int = 4) -> str:
    tf, page_df, _ = _term_stats(pages)
    weight = {s: c * math.log(1 + page_df[s]) for s, c in tf.items()}
    cands = []  # (page, order, sentence)
    order = 0
    for pno, text in enumerate(pages, start=1):
        for s in split_sentences(text):
            if _good_sentence(s):
                cands.append((pno, order, s))
            order += 1
    if not cands:
        return "No summary could be generated from this document's text."

    def score(s: str) -> float:
        toks = [stem(m.group().lower()) for m in _TOKEN_RE.finditer(s) if m.group().lower() not in STOPWORDS]
        if not toks:
            return 0.0
        return sum(weight.get(t, 0.0) for t in set(toks)) / math.sqrt(len(toks))

    defs = [(p, o, _trim_caption_prefix(t)) for p, o, t in cands if p <= 2 and _DEF_VERB.search(t[:160])]
    # the opening definition is usually the best sentence; pick the most on-topic one
    lead = [max(defs, key=lambda c: score(c[2]))] if defs else []
    chosen = list(lead)
    cands = [c for c in cands if not (lead and (c[2] in lead[0][2] or lead[0][2] in c[2]))]  # no near-duplicate of the lead
    used_pages = {c[0] for c in chosen}
    for c in sorted((c for c in cands if c not in chosen), key=lambda c: -score(c[2])):
        if len(chosen) >= len(lead) + n_body:
            break
        if any(u[0] == c[0] for u in chosen) and len(used_pages) < 4:
            continue  # spread across pages
        chosen.append(c)
        used_pages.add(c[0])
    chosen.sort(key=lambda c: c[1])
    return "\n".join(f"• {s} (p.{p})" for p, _, s in chosen)


def llm_summary(pages: list[str], mode: str | None = None) -> str:
    # front-load the start of the document, then sample evenly across the rest
    head = "\n".join(pages[:3])[: MAX_LLM_CHARS // 2]
    rest = pages[3:]
    budget = MAX_LLM_CHARS - len(head)
    per = max(300, budget // max(1, len(rest))) if rest else 0
    sampled = "\n".join(f"[page {i + 4}] {p[:per]}" for i, p in enumerate(rest))[:budget]
    body = f"[pages 1-3] {head}\n{sampled}"
    system = (
        "You summarise documents for a defence-technology analyst. Use ONLY the text provided. "
        "Do not add outside knowledge. Output: one short overview paragraph (max 60 words), then 4-5 bullet "
        "points (start each with '• ') covering the most important points. No preamble."
    )
    return llm.complete(system, [{"role": "user", "content": f"Summarise this document:\n\n{body}"}], max_tokens=500, mode=mode)


def summarise(pages: list[str], mode: str | None = "auto") -> tuple[str, str]:
    """Returns (summary, mode). Falls back to extractive on any LLM failure.
    Raises llm.NotConfigured if the caller explicitly asked for an unconfigured provider."""
    m = llm.resolve(mode)
    if m != "local":
        try:
            return llm_summary(pages, m), m
        except llm.LLMError:
            pass
    return extractive_summary(pages), "extractive"
