"""
BM25 Okapi ranking. Pure Python, no dependencies, no API.

Chosen over TF-IDF cosine because BM25 saturates term frequency (a word
appearing 20 times is not 20x more relevant than once) and normalises for
document length, which matters when chunks vary in size. It is the standard
lexical baseline that dense retrievers are measured against.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for",
    "from", "how", "i", "in", "is", "it", "many", "me", "my", "of", "on",
    "or", "the", "to", "what", "when", "where", "which", "with", "you",
    "your",
}

K1 = 1.5   # term-frequency saturation
B = 0.75   # length normalisation


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower())
            if len(t) > 1 and t not in STOPWORDS]


class BM25Index:
    """In-memory BM25 index over a list of {'source','text'} chunks."""

    def __init__(self) -> None:
        self.chunks: list[dict] = []
        self.tf: list[Counter] = []
        self.lengths: list[int] = []
        self.df: Counter = Counter()
        self.avgdl: float = 0.0

    @property
    def size(self) -> int:
        return len(self.chunks)

    def fit(self, chunks: list[dict]) -> None:
        self.chunks = list(chunks)
        self.tf = []
        self.lengths = []
        self.df = Counter()
        for c in self.chunks:
            toks = tokenize(c["text"])
            counts = Counter(toks)
            self.tf.append(counts)
            self.lengths.append(len(toks))
            self.df.update(counts.keys())
        self.avgdl = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0

    def idf(self, term: str) -> float:
        n = len(self.chunks)
        df = self.df.get(term, 0)
        if df == 0:
            return 0.0
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def score(self, query_tokens: list[str], i: int) -> float:
        tf, dl = self.tf[i], self.lengths[i]
        total = 0.0
        for term in query_tokens:
            f = tf.get(term, 0)
            if not f:
                continue
            denom = f + K1 * (1 - B + B * (dl / self.avgdl if self.avgdl else 1))
            total += self.idf(term) * (f * (K1 + 1)) / denom
        return total

    def search(self, query: str, top_k: int = 3,
               min_score: float = 0.1) -> list[dict]:
        """Return the top-k chunks whose BM25 score clears min_score."""
        qt = tokenize(query)
        if not qt or not self.chunks:
            return []
        scored = [
            {"source": c["source"], "text": c["text"],
             "score": round(self.score(qt, i), 4)}
            for i, c in enumerate(self.chunks)
        ]
        scored.sort(key=lambda r: r["score"], reverse=True)
        return [r for r in scored if r["score"] > min_score][:top_k]
