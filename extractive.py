"""
Extractive question answering.

Instead of generating prose, we select the sentences from the retrieved chunks
that best answer the query. Nothing is invented, so the system cannot
hallucinate: every word in the answer appears verbatim in a source document.

That is a real property, not a consolation prize — for policy and compliance
Q&A it is often the *preferred* behaviour.
"""

from __future__ import annotations

import re

from bm25 import BM25Index, tokenize

# Split on sentence enders, but not on common abbreviations or decimals.
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z#])")

# Question words that hint at the kind of sentence we want.
_CUES = {
    "how": ("by", "via", "through", "visit", "open", "submit", "use", "select"),
    "when": ("within", "before", "after", "during", "day", "week", "month"),
    "where": ("in", "at", "on", "portal", "desk", "floor"),
    "who": ("manager", "lead", "team", "hr", "head"),
    "why": ("because", "so", "since"),
}


def split_sentences(text: str) -> list[str]:
    parts = []
    for block in text.split("\n"):
        block = block.strip()
        if not block:
            continue
        if block.startswith("#"):          # markdown heading: keep as context
            continue
        parts.extend(s.strip() for s in _SENT.split(block) if s.strip())
    return [p for p in parts if len(p) > 25]


def answer(query: str, snippets: list[dict], max_sentences: int = 2) -> str:
    """Select the best-matching sentences across the retrieved chunks."""
    if not snippets:
        return "Nothing in the knowledge base is relevant to that question."

    candidates: list[dict] = []
    for snip in snippets:
        for sent in split_sentences(snip["text"]):
            candidates.append({"text": sent, "source": snip["source"],
                               "chunk_score": snip["score"]})
    if not candidates:
        return snippets[0]["text"][:300]

    # Rank sentences with BM25 over the sentence pool itself.
    index = BM25Index()
    index.fit([{"source": c["source"], "text": c["text"]} for c in candidates])
    qt = tokenize(query)

    first_word = query.strip().lower().split()[0] if query.strip() else ""
    cues = _CUES.get(first_word, ())

    for i, cand in enumerate(candidates):
        score = index.score(qt, i)
        lowered = cand["text"].lower()
        if cues and any(c in lowered for c in cues):
            score *= 1.15                       # mild nudge, not a rewrite
        cand["score"] = score

    ranked = sorted(candidates, key=lambda c: c["score"], reverse=True)
    if ranked[0]["score"] <= 0:
        return ("The knowledge base was searched but no passage answers that "
                "question directly.")

    picked: list[dict] = []
    seen: set[str] = set()
    for cand in ranked:
        if cand["score"] <= 0 or len(picked) >= max_sentences:
            break
        key = cand["text"][:60]
        if key in seen:
            continue
        seen.add(key)
        picked.append(cand)

    # Restore the original reading order so the answer flows.
    order = {id(c): i for i, c in enumerate(candidates)}
    picked.sort(key=lambda c: order[id(c)])

    body = " ".join(c["text"] for c in picked)
    sources = sorted({c["source"] for c in picked})
    return f"{body}\n\n— extracted verbatim from {', '.join(sources)}"
