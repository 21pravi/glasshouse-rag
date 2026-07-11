"""
One retrieval path, two answer paths.

Retrieval is always BM25 — free, deterministic, no key. Both the offline and the
online answer are generated from *the same retrieved chunks*, so a side-by-side
comparison isolates a single variable: extraction versus generation.

    offline : sentences selected verbatim from the chunks  (no model, no key)
    online  : an LLM writes prose grounded in the same chunks (key required)
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import bm25
import extractive
import corpus

RAG_SYSTEM = (
    "You are a helpful assistant. Answer the user's question using ONLY the "
    "provided context. If the context does not contain enough information, say "
    "so plainly. Be concise and name the source document you relied on."
)


class OnlineUnavailable(RuntimeError):
    """Raised when an online answer is requested but no provider is configured."""


class Engine:
    def __init__(self, db_path: str, top_k: int = 3, cache_max: int = 100,
                 chat=None, online_vision=None, offline_vision=None):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hash TEXT UNIQUE, source TEXT, text TEXT)""")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS query_cache (
                query_hash TEXT PRIMARY KEY, result TEXT)""")
        self.conn.commit()

        self.index_ = bm25.BM25Index()
        self.top_k = top_k
        self.cache_max = cache_max
        self.chat = chat                      # None -> online unavailable
        self.online_vision = online_vision
        self.offline_vision = offline_vision

    # -- capability --------------------------------------------------------

    @property
    def online_available(self) -> bool:
        return self.chat is not None

    # -- indexing ----------------------------------------------------------

    def index(self, kb_dir: Path, chunk_size: int = 500,
              overlap: int = 100) -> dict:
        docs = corpus.load_documents(Path(kb_dir), chunk_size, overlap)
        self.index_.fit([{"source": d["source"], "text": d["text"]} for d in docs])

        self.conn.execute("DELETE FROM chunks")
        self.conn.execute("DELETE FROM query_cache")
        for d in docs:
            self.conn.execute(
                "INSERT OR IGNORE INTO chunks (hash, source, text) VALUES (?,?,?)",
                (d["hash"], d["source"], d["text"]))
        self.conn.commit()
        return {"indexed": len(docs), "total_chunks": len(docs),
                "documents": len({d["source"] for d in docs})}

    # -- retrieval ---------------------------------------------------------

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        return self.index_.search(query, top_k=top_k or self.top_k)

    # -- cache (keyed on query + mode) -------------------------------------

    @staticmethod
    def _key(query: str, mode: str) -> str:
        return hashlib.md5(f"{mode}::{query.lower().strip()}".encode()).hexdigest()

    def _cached(self, query: str, mode: str) -> dict | None:
        row = self.conn.execute("SELECT result FROM query_cache WHERE query_hash=?",
                                (self._key(query, mode),)).fetchone()
        return json.loads(row[0]) if row else None

    def _cache(self, query: str, mode: str, result: dict) -> None:
        n = self.conn.execute("SELECT COUNT(*) FROM query_cache").fetchone()[0]
        if n >= self.cache_max:
            self.conn.execute("DELETE FROM query_cache WHERE rowid IN "
                              "(SELECT rowid FROM query_cache LIMIT 10)")
        self.conn.execute("INSERT OR REPLACE INTO query_cache VALUES (?,?)",
                          (self._key(query, mode), json.dumps(result)))
        self.conn.commit()

    def clear_cache(self) -> None:
        self.conn.execute("DELETE FROM query_cache")
        self.conn.commit()

    # -- answers -----------------------------------------------------------

    def answer_offline(self, query: str, snippets: list[dict]) -> str:
        return extractive.answer(query, snippets)

    def answer_online(self, query: str, snippets: list[dict]) -> str:
        if not self.online_available:
            raise OnlineUnavailable(
                "No API key configured, so the online answer is unavailable. "
                "The offline answer needs no key.")
        context = "\n\n---\n\n".join(
            f"[Source: {s['source']}]\n{s['text']}" for s in snippets)
        return self.chat.complete(
            RAG_SYSTEM,
            [{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}])

    def ask(self, query: str, mode: str = "both") -> dict:
        if mode not in ("offline", "online", "both"):
            raise ValueError(f"unknown mode: {mode}")

        cached = self._cached(query, mode)
        if cached:
            return {**cached, "from_cache": True}

        snippets = self.search(query)
        payload = {
            "query": query, "metric": "bm25", "from_cache": False,
            "snippets": [{"source": s["source"], "text": s["text"][:220],
                          "score": s["score"]} for s in snippets],
            "sources": list(dict.fromkeys(s["source"] for s in snippets)),
            "offline": None, "online": None, "online_error": None,
            "online_available": self.online_available,
        }

        if not snippets:
            nothing = "Nothing in the knowledge base is relevant to that question."
            if mode in ("offline", "both"):
                payload["offline"] = nothing
            if mode in ("online", "both"):
                payload["online"] = nothing
            self._cache(query, mode, payload)
            return payload

        if mode in ("offline", "both"):
            payload["offline"] = self.answer_offline(query, snippets)

        if mode in ("online", "both"):
            try:
                payload["online"] = self.answer_online(query, snippets)
            except OnlineUnavailable as exc:
                payload["online_error"] = str(exc)
            except Exception as exc:                      # provider/network failure
                payload["online_error"] = f"Model call failed: {exc}"

        # Never cache a transient online failure.
        if payload["online_error"] is None:
            self._cache(query, mode, payload)
        return payload

    # -- vision ------------------------------------------------------------

    def describe(self, image_bytes: bytes, mime_type: str, mode: str = "both") -> dict:
        out = {"offline": None, "online": None, "online_error": None,
               "online_available": self.online_vision is not None}

        if mode in ("offline", "both"):
            import vision as vision_parse
            out["offline"] = vision_parse._parse_vision_response(
                self.offline_vision.describe(image_bytes, mime_type))

        if mode in ("online", "both"):
            if self.online_vision is None:
                out["online_error"] = ("No API key configured, so the model "
                                       "caption is unavailable.")
            else:
                try:
                    import vision as vision_parse
                    out["online"] = vision_parse._parse_vision_response(
                        self.online_vision.describe(image_bytes, mime_type))
                except Exception as exc:
                    out["online_error"] = f"Vision call failed: {exc}"
        return out

    # -- stats -------------------------------------------------------------

    def stats(self) -> dict:
        total = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        by_source = self.conn.execute(
            "SELECT source, COUNT(*) FROM chunks GROUP BY source ORDER BY source"
        ).fetchall()
        cached = self.conn.execute("SELECT COUNT(*) FROM query_cache").fetchone()[0]
        return {"total_chunks": total, "cached_queries": cached,
                "documents": [{"source": s, "chunks": n} for s, n in by_source],
                "online_available": self.online_available}
