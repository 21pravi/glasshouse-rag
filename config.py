"""
Central configuration, loaded from environment variables with sane defaults.

Every other module imports this. Values are read once at import time; the
Streamlit demo (app.py) overrides the module-level attributes at runtime.
"""

import os
from pathlib import Path

# --- Secrets (required at runtime, not at import time) ---------------------
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

# --- Models ----------------------------------------------------------------
OPENAI_CHAT_MODEL: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL",
                                        "text-embedding-3-small")

# --- RAG pipeline ----------------------------------------------------------
KNOWLEDGE_BASE_DIR: Path = Path(os.getenv("KNOWLEDGE_BASE_DIR", "knowledge_base"))
SQLITE_DB_PATH: str = os.getenv("SQLITE_DB_PATH", "vectorstore.db")
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K: int = int(os.getenv("TOP_K", "3"))

# --- Bot behaviour ---------------------------------------------------------
MAX_HISTORY_PER_USER: int = int(os.getenv("MAX_HISTORY_PER_USER", "3"))
CACHE_MAX_SIZE: int = int(os.getenv("CACHE_MAX_SIZE", "100"))


def require(name: str) -> str:
    """Fetch a required secret, raising a clear error if it is unset."""
    value = globals().get(name, "")
    if not value:
        raise RuntimeError(
            f"{name} is not set. Add it to your .env file or export it "
            f"before running."
        )
    return value
