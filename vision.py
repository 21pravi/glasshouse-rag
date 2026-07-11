"""
Parsing helpers for vision output.

Both the classical-CV analyser (`imageanalysis.py`) and the online vision
providers emit the same CAPTION / TAGS / DETAILS block, so the UI and the bot
can render either without knowing which produced it.
"""

from __future__ import annotations

from pathlib import Path

MIME_TYPES = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
}


def get_image_mime_type(path: str) -> str:
    """Best-effort MIME type from a filename; JPEG when unknown."""
    return MIME_TYPES.get(Path(path).suffix.lower(), "image/jpeg")


def _parse_vision_response(raw: str) -> dict:
    """Parse a CAPTION / TAGS / DETAILS block, tolerating malformed output."""
    out = {"caption": "", "tags": [], "detailed": ""}
    for line in raw.split("\n"):
        line = line.strip()
        if line.lower().startswith("caption:"):
            out["caption"] = line.split(":", 1)[1].strip()
        elif line.lower().startswith("tags:"):
            out["tags"] = [t.strip() for t in line.split(":", 1)[1].split(",") if t.strip()]
        elif line.lower().startswith("details:"):
            out["detailed"] = line.split(":", 1)[1].strip()

    if not out["caption"]:
        out["caption"] = raw.strip()[:120]
    if not out["tags"]:
        out["tags"] = ["image", "uploaded", "analysis"]
    return out
