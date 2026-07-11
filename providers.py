"""
Pluggable embedding / chat / vision providers.

Three modes, chosen by whichever API key is present:

  openai  — OpenAI embeddings + gpt-4o-mini chat and vision
  claude  — local TF-IDF embeddings + Claude chat and vision
             (Anthropic has no embeddings endpoint, so retrieval runs locally)
  demo    — local TF-IDF embeddings + extractive answers, no API calls at all

Demo mode exists so a public portfolio URL can show real retrieval without
exposing an API key to every visitor.
"""

from __future__ import annotations

import base64
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Protocol

# ---------------------------------------------------------------------------
# Interfaces
# ---------------------------------------------------------------------------


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class Chat(Protocol):
    def complete(self, system: str, messages: list[dict]) -> str: ...


class Vision(Protocol):
    def describe(self, image_bytes: bytes, mime_type: str) -> str: ...


# ---------------------------------------------------------------------------
# Local TF-IDF embedder (no API, no cost)
# ---------------------------------------------------------------------------

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "for", "from", "in",
    "is", "it", "of", "on", "or", "the", "to", "with", "you", "your", "i",
}


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]


class LocalTfidfEmbedder:
    """Classic TF-IDF with L2 normalisation. Cosine similarity then behaves
    exactly as it does for dense embeddings, so the rest of the pipeline is
    unchanged."""

    def __init__(self, state_path: str | Path | None = None):
        self.vocab: dict[str, int] = {}
        self.idf: list[float] = []
        self.state_path = Path(state_path) if state_path else None
        if self.state_path and self.state_path.exists():
            self._load()

    @property
    def fitted(self) -> bool:
        return bool(self.vocab)

    def fit(self, corpus: list[str]) -> None:
        df: Counter[str] = Counter()
        for doc in corpus:
            df.update(set(_tokens(doc)))
        self.vocab = {tok: i for i, tok in enumerate(sorted(df))}
        n = len(corpus)
        self.idf = [0.0] * len(self.vocab)
        for tok, i in self.vocab.items():
            self.idf[i] = math.log((1 + n) / (1 + df[tok])) + 1.0
        if self.state_path:
            self._save()

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * len(self.vocab)
        counts = Counter(_tokens(text))
        for tok, c in counts.items():
            idx = self.vocab.get(tok)
            if idx is not None:
                vec[idx] = c * self.idf[idx]
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else vec

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.fitted:
            raise RuntimeError("LocalTfidfEmbedder.fit() must run before embed()")
        return [self._vector(t) for t in texts]

    def _save(self) -> None:
        self.state_path.write_text(json.dumps({"vocab": self.vocab, "idf": self.idf}))

    def _load(self) -> None:
        data = json.loads(self.state_path.read_text())
        self.vocab, self.idf = data["vocab"], data["idf"]


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


class OpenAIEmbedder:
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(input=texts, model=self.model)
        return [d.embedding for d in resp.data]


class OpenAIChat:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def complete(self, system: str, messages: list[dict]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, *messages],
            temperature=0.3, max_tokens=500,
        )
        return resp.choices[0].message.content.strip()


class OpenAIVision:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def describe(self, image_bytes: bytes, mime_type: str) -> str:
        b64 = base64.b64encode(image_bytes).decode()
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": VISION_SYSTEM},
                {"role": "user", "content": [
                    {"type": "text", "text": "Describe this image."},
                    {"type": "image_url",
                     "image_url": {"url": f"data:{mime_type};base64,{b64}",
                                   "detail": "low"}},
                ]},
            ],
            temperature=0.3, max_tokens=300,
        )
        return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Anthropic / Claude
# ---------------------------------------------------------------------------


class AnthropicChat:
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6"):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def complete(self, system: str, messages: list[dict]) -> str:
        resp = self.client.messages.create(
            model=self.model, system=system, messages=messages,
            temperature=0.3, max_tokens=500,
        )
        return "".join(b.text for b in resp.content if b.type == "text").strip()


class AnthropicVision:
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6"):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def describe(self, image_bytes: bytes, mime_type: str) -> str:
        b64 = base64.b64encode(image_bytes).decode()
        resp = self.client.messages.create(
            model=self.model, system=VISION_SYSTEM, temperature=0.3, max_tokens=300,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64",
                                             "media_type": mime_type, "data": b64}},
                {"type": "text", "text": "Describe this image."},
            ]}],
        )
        return "".join(b.text for b in resp.content if b.type == "text").strip()


# ---------------------------------------------------------------------------
# Demo (no network)
# ---------------------------------------------------------------------------

VISION_SYSTEM = (
    "You are an image analysis assistant. Respond in exactly this format:\n"
    "CAPTION: <one-sentence description>\n"
    "TAGS: <tag1>, <tag2>, <tag3>\n"
    "DETAILS: <2-3 sentence detailed description>"
)


VISION_SYSTEM = (
    "You are an image analysis assistant. Respond in exactly this format:\n"
    "CAPTION: <one-sentence description>\n"
    "TAGS: <tag1>, <tag2>, <tag3>\n"
    "DETAILS: <2-3 sentence detailed description>"
)


class OfflineChat:
    """Extractive QA. Selects sentences from the retrieved chunks; invents
    nothing, so it cannot hallucinate."""

    def complete(self, system: str, messages: list[dict]) -> str:  # unused args
        raise NotImplementedError("OfflineChat is driven by Engine.ask directly")


class OfflineVision:
    """Classical computer vision. No model, no key."""

    def describe(self, image_bytes: bytes, mime_type: str) -> str:
        import imageanalysis
        return imageanalysis.describe(image_bytes)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_providers(openai_key: str = "", anthropic_key: str = "",
                    state_path: str | Path | None = None
                    ) -> tuple[str, Embedder | None, Chat | None, Vision]:
    """Return (mode, embedder, chat, vision) based on available credentials.

    With no keys we fall back to `offline`: BM25 retrieval, extractive answers,
    classical-CV image analysis. Nothing leaves the machine."""
    if openai_key:
        return "openai", OpenAIEmbedder(openai_key), OpenAIChat(openai_key), \
            OpenAIVision(openai_key)
    if anthropic_key:
        return "claude", LocalTfidfEmbedder(state_path), \
            AnthropicChat(anthropic_key), AnthropicVision(anthropic_key)
    return "offline", None, OfflineChat(), OfflineVision()
