"""Text utilities: tokenising, light stemming, sentence splitting, BM25.

Pure Python on purpose: the retrieval layer is small enough to explain line by
line in the project video, and has no model download or API dependency.
"""
from __future__ import annotations

import math
import re
from collections import Counter

STOPWORDS = set(
    """a about above after again against all am an and any are as at be because been
    before being below between both but by can could did do does doing down during
    each few for from further had has have having he her here hers him his how i if
    in into is it its itself just me more most my no nor not of off on once only or
    other our out over own same she should so some such than that the their theirs
    them then there these they this those through to too under until up very was we
    were what when where which while who whom why will with would you your also may
    might must shall upon via per et al""".split()
)

# Words that describe the *question*, not the content being asked about.
META_WORDS = set(
    """document documents doc paper article text pdf report according discussed discuss
    mentioned mention described describe describes say says said state states stated
    tell give name list explain major main key one many much use used using page pages
    section sections provided provide present within across following""".split()
)

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9']*|\d+(?:[.,]\d+)*")  # hyphens split: "radar-guided" -> radar, guided
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")


def stem(word: str) -> str:
    """Very small suffix stripper. Consistency matters more than linguistics."""
    w = word.lower().strip("'-")
    if w.endswith("'s"):
        w = w[:-2]
    if len(w) <= 3 or w.isdigit():
        return w
    for suf, rep in (("ies", "y"), ("sses", "ss"), ("ations", "ate"), ("ation", "ate"),
                     ("ingly", ""), ("edly", ""), ("ing", ""), ("ness", ""), ("ment", ""),
                     ("ed", ""), ("ly", ""), ("es", ""), ("s", "")):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            if suf == "s" and w.endswith("ss"):
                break
            w = w[: -len(suf)] + rep
            break
    if len(w) > 4 and w[-1] == w[-2] and w[-1] not in "ls":  # jamm -> jam
        w = w[:-1]
    return w


def tokenize(text: str, keep_stop: bool = False) -> list[str]:
    toks = []
    for m in _TOKEN_RE.finditer(text):
        t = m.group().lower()
        if not keep_stop and t in STOPWORDS:
            continue
        toks.append(stem(t))
    return toks


def tokenize_spans(text: str):
    """Yield (stem, start, end) for every non-stopword token — used for highlighting."""
    for m in _TOKEN_RE.finditer(text):
        t = m.group().lower()
        if t in STOPWORDS:
            continue
        yield stem(t), m.start(), m.end()


def query_terms(question: str) -> list[str]:
    """Content stems of a question, minus stopwords and question-meta words."""
    out = []
    for m in _TOKEN_RE.finditer(question):
        t = m.group().lower()
        if t in STOPWORDS or t in META_WORDS:
            continue
        s = stem(t)
        if s not in out:
            out.append(s)
    return out


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [s.strip() for s in _SENT_SPLIT.split(text) if len(s.strip()) > 1]


class BM25:
    """Okapi BM25 over a list of documents (each a string)."""

    def __init__(self, texts: list[str], k1: float = 1.4, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg = (sum(self.lens) / len(self.lens)) if self.lens else 0.0
        df: Counter = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.n = n
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def idf_of(self, term: str) -> float:
        # unseen term: treat as maximally informative
        return self.idf.get(term, math.log(1 + (self.n + 0.5) / 0.5))

    def score(self, terms: list[str], i: int) -> float:
        d, ln = self.docs[i], self.lens[i]
        s = 0.0
        for t in terms:
            f = d.get(t, 0)
            if not f:
                continue
            denom = f + self.k1 * (1 - self.b + self.b * ln / (self.avg or 1))
            s += self.idf.get(t, 0.0) * f * (self.k1 + 1) / denom
        return s

    def rank(self, terms: list[str], top_k: int = 6) -> list[tuple[int, float]]:
        scored = [(i, self.score(terms, i)) for i in range(self.n)]
        scored = [x for x in scored if x[1] > 0]
        scored.sort(key=lambda x: (-x[1], x[0]))  # deterministic tie-break
        return scored[:top_k]


def proper_terms(question: str) -> list[str]:
    """Stems of capitalised / ALLCAPS words after the first word (names, acronyms, places)."""
    out = []
    for n, m in enumerate(_TOKEN_RE.finditer(question)):
        raw = m.group()
        if n and raw[0].isupper() and raw.lower() not in STOPWORDS | META_WORDS:
            out.append(stem(raw.lower()))
    return out
