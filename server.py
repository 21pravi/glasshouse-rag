"""
FastAPI server. One UI, two answer engines.

    (no key)             -> offline only; the online panel explains why
    OPENAI_API_KEY set   -> online answers from gpt-4o-mini
    ANTHROPIC_API_KEY    -> online answers from Claude

Run:    uvicorn server:app --reload
Deploy: uvicorn server:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import providers
from engine import Engine

ROOT = Path(__file__).resolve().parent
KB_DIR = Path(os.getenv("KNOWLEDGE_BASE_DIR", ROOT / "knowledge_base"))
DB_PATH = os.getenv("SQLITE_DB_PATH", str(ROOT / "index.db"))
MAX_IMAGE_BYTES = 5 * 1024 * 1024
ALLOWED_MIME = {"image/png", "image/jpeg", "image/gif", "image/webp", "image/bmp"}

Mode = Literal["offline", "online", "both"]
state: dict = {}


def _online_providers():
    """Return (name, chat, vision) or (None, None, None) if no key is present.

    Any of OPENAI_API_KEY, ANTHROPIC_API_KEY, GROQ_API_KEY, GEMINI_API_KEY,
    XAI_API_KEY, DEEPSEEK_API_KEY or MISTRAL_API_KEY enables the online engine.
    See providers.PROVIDERS for the full list and default models.
    """
    return providers.build_online_providers()


@asynccontextmanager
async def lifespan(app: FastAPI):
    name, chat, online_vision = _online_providers()
    engine = Engine(DB_PATH, top_k=int(os.getenv("TOP_K", "3")),
                    chat=chat, online_vision=online_vision,
                    offline_vision=providers.OfflineVision())
    engine.index(KB_DIR,
                 chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
                 overlap=int(os.getenv("CHUNK_OVERLAP", "100")))
    state.update(engine=engine, online_provider=name)
    yield
    engine.conn.close()


app = FastAPI(title="RAG pipeline inspector — offline vs online", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    mode: Mode = "both"


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/healthz")
def healthz() -> dict:
    """Liveness probe for the host (Render, Fly, etc).

    Deliberately does not touch engine state: it returns 200 as soon as the
    process is up, so the platform's health check passes even during the brief
    window before indexing finishes.
    """
    return {"status": "ok"}


@app.get("/api/health")
def health() -> dict:
    engine = state.get("engine")
    if engine is None:
        return {"status": "starting", "online_available": False,
                "online_provider": None}
    return {"status": "ok",
            "online_available": engine.online_available,
            "online_provider": state.get("online_provider")}


@app.get("/api/stats")
def stats() -> dict:
    return {**state["engine"].stats(),
            "online_provider": state["online_provider"]}


@app.post("/api/ask")
def ask(req: AskRequest) -> dict:
    return {**state["engine"].ask(req.query.strip(), req.mode),
            "online_provider": state["online_provider"]}


@app.post("/api/describe")
async def describe(file: UploadFile = File(...), mode: str = Form("both")) -> dict:
    if mode not in ("offline", "online", "both"):
        raise HTTPException(status_code=422, detail=f"unknown mode: {mode}")
    if file.content_type not in ALLOWED_MIME:
        raise HTTPException(status_code=415,
                            detail=f"Unsupported image type: {file.content_type}")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds 5 MB.")
    return {**state["engine"].describe(data, file.content_type, mode),
            "online_provider": state["online_provider"]}


@app.post("/api/cache/clear")
def clear_cache() -> dict:
    state["engine"].clear_cache()
    return {"cleared": True}
