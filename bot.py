"""
Telegram front-end for the Glasshouse engine.

This is a thin adapter. All retrieval, answering and image analysis happen in
`engine.Engine` — the same object `server.py` uses — so the bot and the web UI
cannot drift apart.

    /ask <query>   retrieve, then answer (offline, online, or both)
    /image         send a photo to have it analysed
    /summarize     recap the last interaction
    /mode          show or change the answer engine
    /help, /start

Runs with no API key: the offline engine answers, and /mode explains why the
online engine is idle.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

import config
import extractive
import providers
from engine import Engine
from history import HistoryManager

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

VALID_MODES = ("offline", "online", "both")

engine: Engine | None = None
history_mgr = HistoryManager()
_default_mode = "both"


# ---------------------------------------------------------------------------
# Pure helpers — no Telegram types, so they are directly testable
# ---------------------------------------------------------------------------

def build_engine(db_path: str | None = None) -> tuple[Engine, str | None]:
    """Construct the shared engine, mirroring server.py's provider selection.

    The online engine turns on for any supported key (OpenAI, Anthropic, Groq,
    Gemini, xAI/Grok, DeepSeek, Mistral); see providers.PROVIDERS.
    """
    name, chat, online_vision = providers.build_online_providers()

    eng = Engine(db_path or os.getenv("SQLITE_DB_PATH", "index.db"),
                 top_k=int(os.getenv("TOP_K", "3")),
                 chat=chat, online_vision=online_vision,
                 offline_vision=providers.OfflineVision())
    eng.index(Path(os.getenv("KNOWLEDGE_BASE_DIR", "knowledge_base")))
    return eng, name


def resolve_mode(requested: str | None, online_available: bool) -> str:
    """Fall back to offline whenever the online engine cannot run."""
    mode = (requested or "both").lower()
    if mode not in VALID_MODES:
        mode = "both"
    if not online_available and mode in ("online", "both"):
        return "offline"
    return mode


def format_sources(snippets: list[dict], metric: str = "bm25") -> str:
    """Render the retrieval trace.

    BM25 scores are unbounded, so they are shown raw. Presenting `score * 100`
    as a percentage — as the earlier cosine-only version did — would report a
    BM25 score of 7.57 as "757% relevance".
    """
    if not snippets:
        return ""
    lines = [f"\nSources ({metric} score):"]
    for i, s in enumerate(snippets, start=1):
        preview = s["text"][:110].replace("\n", " ").strip()
        lines.append(f"  {i}. {s['source']} - {s['score']:.2f}")
        lines.append(f"     {preview}...")
    return "\n".join(lines)


def format_answer(result: dict, mode: str) -> str:
    """Compose the reply text for /ask."""
    parts = []
    if result.get("from_cache"):
        parts.append("(cached)")

    if mode in ("offline", "both") and result.get("offline"):
        header = "Answer - extractive" if mode == "both" else "Answer"
        parts.append(f"{header}\n{result['offline']}")

    if mode in ("online", "both"):
        if result.get("online"):
            header = "Answer - generated" if mode == "both" else "Answer"
            parts.append(f"{header}\n{result['online']}")
        elif result.get("online_error"):
            parts.append(f"Online engine unavailable: {result['online_error']}")

    trace = format_sources(result.get("snippets", []), result.get("metric", "score"))
    if trace:
        parts.append(trace)
    return "\n\n".join(parts)


def format_vision(result: dict, mode: str) -> str:
    """Compose the reply text for a photo."""
    parts = []

    def block(title: str, v: dict) -> str:
        return (f"{title}\nCaption: {v['caption']}\n"
                f"Tags: {', '.join(v['tags'])}\n"
                f"Details: {v.get('detailed', '-')}")

    if mode in ("offline", "both") and result.get("offline"):
        parts.append(block("Image analysis - classical CV" if mode == "both"
                           else "Image analysis", result["offline"]))
    if mode in ("online", "both"):
        if result.get("online"):
            parts.append(block("Image analysis - vision model" if mode == "both"
                               else "Image analysis", result["online"]))
        elif result.get("online_error"):
            parts.append(f"Online engine unavailable: {result['online_error']}")
    return "\n\n".join(parts)


def summarize_offline(text: str, max_sentences: int = 2) -> str:
    """Deterministic recap: the first sentences of the last reply, verbatim.

    No model, so it works without a key - and, like the extractive answer, it
    cannot invent anything that was not already said.
    """
    sentences = extractive.split_sentences(text)
    if not sentences:
        return text.strip()[:280]
    return " ".join(sentences[:max_sentences])


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

async def start_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Glasshouse - a RAG pipeline you can see through.\n\n"
        "/ask <question>   search the knowledge base\n"
        "/image            send a photo to have it analysed\n"
        "/summarize        recap the last interaction\n"
        "/mode             switch answer engine\n"
        "/help             detailed usage\n\n"
        "Try: /ask What is the remote work policy?")


async def help_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    online = engine.online_available if engine else False
    await update.message.reply_text(
        "Two answer engines share one retrieval path (BM25).\n\n"
        "offline - sentences selected verbatim from the sources. No key, no "
        "model, cannot hallucinate.\n"
        "online  - an LLM writes prose from the same chunks. Needs an API key.\n"
        "both    - shows each, side by side.\n\n"
        f"Online engine: {'available' if online else 'idle (no API key set)'}\n\n"
        "Knowledge base: company policies, technical FAQs, onboarding guide, "
        "product information, recipes.\n\n"
        "Commands: /ask, /image, /summarize, /mode [offline|online|both]\n"
        "I keep the last 3 exchanges for context.")


async def mode_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    global _default_mode
    if not ctx.args:
        await update.message.reply_text(
            f"Current mode: {_default_mode}\n"
            f"Online engine: {'available' if engine.online_available else 'idle'}\n"
            "Change with /mode offline | online | both")
        return

    requested = ctx.args[0].lower()
    if requested not in VALID_MODES:
        await update.message.reply_text(
            f"Unknown mode '{requested}'. Choose offline, online or both.")
        return
    if requested in ("online", "both") and not engine.online_available:
        await update.message.reply_text(
            f"Cannot switch to '{requested}': no API key is configured. "
            "Staying on offline, which needs none.")
        return

    _default_mode = requested
    await update.message.reply_text(f"Mode set to {requested}.")


async def ask_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not ctx.args:
        await update.message.reply_text(
            "Please provide a question.\nUsage: /ask What is the leave policy?")
        return

    query = " ".join(ctx.args)
    user_id = update.effective_user.id
    mode = resolve_mode(_default_mode, engine.online_available)

    await update.message.reply_text("Searching the knowledge base...")
    try:
        result = engine.ask(query, mode)
        reply = format_answer(result, mode)
        history_mgr.add(user_id, "user", query, "ask")
        history_mgr.add(user_id, "assistant",
                        result.get("offline") or result.get("online") or "", "ask")
        await update.message.reply_text(reply)
    except Exception as exc:
        logger.error("Error in /ask: %s", exc, exc_info=True)
        await update.message.reply_text(f"Sorry, something went wrong: {str(exc)[:200]}")


async def image_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Send me a photo and I'll analyse it. The /image command is optional.")


async def handle_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    mode = resolve_mode(_default_mode, engine.online_available)
    await update.message.reply_text("Analysing the image...")

    tmp_path = None
    try:
        photo = update.message.photo[-1]
        file = await ctx.bot.get_file(photo.file_id)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
        await file.download_to_drive(tmp_path)

        data = Path(tmp_path).read_bytes()
        result = engine.describe(data, "image/jpeg", mode)
        reply = format_vision(result, mode)

        history_mgr.add(user_id, "user", "[uploaded an image]", "image")
        history_mgr.add(user_id, "assistant", reply, "image")
        await update.message.reply_text(reply)
    except Exception as exc:
        logger.error("Error in photo handler: %s", exc, exc_info=True)
        await update.message.reply_text(f"Couldn't analyse the image: {str(exc)[:200]}")
    finally:
        # The original leaked the temp file whenever analysis raised.
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


async def summarize_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    last = history_mgr.get_last_interaction(user_id)
    if not last:
        await update.message.reply_text(
            "Nothing to summarize yet. Try /ask or send a photo first.")
        return

    if not engine.online_available:
        await update.message.reply_text(
            "Summary (extractive):\n" + summarize_offline(last["content"]))
        return

    try:
        summary = engine.chat.complete(
            "Summarize the following bot response in 2-3 concise sentences. "
            "State whether it was a question answer or an image description.",
            [{"role": "user",
              "content": f"Interaction type: {last['type']}\n"
                         f"Response:\n{last['content']}"}])
        await update.message.reply_text(f"Summary:\n{summary}")
    except Exception as exc:
        logger.error("Error in /summarize: %s", exc, exc_info=True)
        # An online failure must not deny the user a summary.
        await update.message.reply_text(
            "Summary (extractive - the online engine failed):\n"
            + summarize_offline(last["content"]))


async def handle_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message and update.message.text:
        ctx.args = update.message.text.split()
        await ask_cmd(update, ctx)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

def build_app() -> Application:
    global engine

    if not config.TELEGRAM_BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN is required to run the bot.")
    # No OPENAI_API_KEY check: the offline engine needs no key.

    logger.info("Building the shared engine...")
    engine, provider = build_engine()
    logger.info("Indexed knowledge base. Online engine: %s",
                provider or "idle (no API key)")

    app = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("ask", ask_cmd))
    app.add_handler(CommandHandler("mode", mode_cmd))
    app.add_handler(CommandHandler("image", image_cmd))
    app.add_handler(CommandHandler("summarize", summarize_cmd))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    return app


def main() -> None:
    app = build_app()
    logger.info("Bot is running. Ctrl+C to stop.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
