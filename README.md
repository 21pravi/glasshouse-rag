<div align="center">

# Glasshouse

**A RAG pipeline you can see through.**

Two answer engines, one retrieval trace. Watch what the retriever actually found,
then compare an answer that *quotes* against an answer that *writes*.

[**Live demo**](https://21pravi.github.io/glasshouse-rag/) · [Architecture](ARCHITECTURE.md) · [Methodology](METHODOLOGY.md)

</div>

---

Most RAG demos are a chat box with an API call behind it. You cannot see what was
retrieved, why it was ranked that way, or what the model was actually given. Glasshouse
puts the retrieval trace front and centre and runs **two answer engines side by side over
identical retrieved context**, so the only variable is how the answer is produced.

|  | Offline engine | Online engine |
|---|---|---|
| Retrieval | BM25 Okapi | BM25 Okapi *(identical)* |
| Answering | extractive — sentences selected verbatim | generative — an LLM writes prose |
| Vision | classical CV: colour quantisation, edge filtering, exposure statistics | vision model |
| Needs an API key | **no** | yes |
| Can hallucinate | **no** | yes |
| Cost per query | zero | per token |

Because retrieval is shared, a difference in the two answers is attributable to the
answering step and nothing else.

## Why extractive answers matter

The offline engine cannot hallucinate. Not "rarely" — *cannot*. It selects whole
sentences from the retrieved chunks and returns them unchanged. A test asserts that
every sentence in an offline answer appears verbatim in a source document.

> **Q:** How do I reset my VPN password?
> **A:** Visit the internal identity portal, select "VPN Credentials", and choose Reset.
> — *extracted verbatim from tech_faqs.md*

For policy, compliance, and HR question-answering, that property is often worth more
than fluency. Glasshouse shows both so you can judge the trade-off yourself.

## Quick start

```bash
git clone https://github.com/21pravi/glasshouse-rag
cd glasshouse-rag

# Offline engine: this is the complete dependency list.
pip install -r requirements.txt
uvicorn server:app --reload          # → http://localhost:8000
```

No API key required. The online panel will explain that it is idle; everything else works.

To enable the online engine, set **one** of:

```bash
pip install -r requirements-online.txt
export OPENAI_API_KEY=...            # → gpt-4o-mini
export ANTHROPIC_API_KEY=...         # → Claude
```

### Deployment

There are two ways to host this, and they differ only in how the *online* engine works.
The offline engine is identical either way.

**Option A — hosted server (recommended, and what the live demo runs).** Deploy the
FastAPI app; set any one supported provider key in the host's dashboard (see below). The browser calls
`/api/ask`, the server makes the Anthropic call server-side, and the key never leaves the
host. No CORS, nothing for a visitor to configure.

```bash
# One-click on Render: New + → Blueprint → point at this repo.
# render.yaml provisions everything; set one provider key in the dashboard.
# Or run the same thing anywhere:
pip install -r requirements-online.txt
export GROQ_API_KEY=...          # free tier; or any provider below
uvicorn server:app --host 0.0.0.0 --port 8000
```

**Supported providers.** Set any one of these; the online engine adapts automatically:

| Env var | Provider | Free tier | Vision |
|---|---|---|---|
| `OPENAI_API_KEY` | OpenAI (gpt-4o-mini) | no | yes |
| `ANTHROPIC_API_KEY` | Claude | no | yes |
| `GROQ_API_KEY` | Groq (Llama 3.3 70B) | **yes** | no |
| `GEMINI_API_KEY` | Google Gemini Flash | **yes** | yes |
| `XAI_API_KEY` | xAI (Grok) | no | yes |
| `DEEPSEEK_API_KEY` | DeepSeek | no | no |
| `MISTRAL_API_KEY` | Mistral | **yes** | no |

Every provider except Anthropic speaks the OpenAI chat-completions protocol, so one
class drives all of them — adding a provider is one row in `providers.PROVIDERS`.

Leave the key unset and the service runs offline-only — the online panel says so.

**Option B — static site, zero backend.** The offline engine runs entirely in the
browser, so the whole app also compiles to one static file:

```bash
python build_static.py               # → docs/index.html   (~44 KB, no server)
```

Push `docs/` and enable GitHub Pages: the offline demo then runs free, forever, with no
backend and no key. In this mode the online panel lets the *visitor* pick a provider and paste their own
key (kept in memory for that tab only, never stored). That browser-direct call
depends on Anthropic's CORS policy, which is why Option A is preferred when you want the
online engine to work reliably for everyone.

## How it works

```
knowledge_base/*.md
        │
        ├─ chunk (500 chars, 100 overlap)
        │
        └─ BM25 Okapi index ──── query ──── ranked chunks + scores
                                                  │
                        ┌─────────────────────────┴─────────────────────────┐
                        │                                                   │
             sentence-level BM25                                LLM, prompted with
             → select top sentences                             the same chunks
                        │                                                   │
              offline answer (verbatim)                     online answer (generated)
```

Full detail in [ARCHITECTURE.md](ARCHITECTURE.md); the maths and the design decisions
are in [METHODOLOGY.md](METHODOLOGY.md).

## Tests

138 tests, none of which need an API key.

```bash
pip install -r requirements-dev.txt
python tests/make_expected.py        # regenerate parity fixtures from source
pytest tests/ -q                     # 93 passed
node tests/test_static_parity.mjs    # 45 passed
```

The browser port of the offline engine is a second implementation of the same
algorithms, so it is tested *against* the Python one: BM25 scores must agree to four
decimal places and extractive answers must match character-for-character. The suite
also asserts that offline mode issues **zero** network calls, that an online outage never
degrades the offline answer, and that no second retrieval engine has crept back into any
module.

The Telegram bot is tested too, with fake `Update`/`Context` objects, so its handlers run
for real without touching Telegram.

## Project layout

| File | Role |
|---|---|
| `bm25.py` | BM25 Okapi ranking. No dependencies. |
| `extractive.py` | Sentence selection — the offline answer engine. |
| `imageanalysis.py` | Classical-CV image description. No model. |
| `corpus.py` | Document loading and chunking. |
| `engine.py` | One retrieval path, two answer paths, per-mode cache. |
| `providers.py` | OpenAI / Anthropic / offline backends behind one interface. |
| `static/` | The UI. Shared by the server and the static build. |
| `build_static.py` | Compiles the UI + offline engine into `docs/index.html`. |
| `server.py` and `bot.py` | Two thin front-ends over one `Engine`. Neither owns any pipeline logic. |

## Honest limits

- The web UI and the Telegram bot run the same `Engine`. Retrieval is BM25 only; the dense
  embedder in `providers.py` is not wired in.
- The offline engine is **not** generative AI. BM25 is a ranking function; extractive QA
  selects rather than writes; the image analysis measures pixels. Nothing is generated.
  Only the online engine is GenAI.
- BM25 is lexical. It will miss a paraphrase that shares no vocabulary with the source.
  A dense-embedding retriever is available in `providers.py` if you have a key.
- The image analysis reports colour, exposure, contrast, edge density and sharpness.
  It identifies no objects and no people, and it says so in its own output.
- Two ways to run the online engine. The hosted server (`server.py`) calls Anthropic
  server-side and is covered by tests against a live uvicorn process. The static build's
  browser-direct call depends on Anthropic's CORS policy and is the one path not exercised
  end-to-end in CI — which is why the hosted server is the recommended deployment.

## Licence

MIT — see [LICENSE](LICENSE).

Built by [Praviveek](https://praviveek.com) · [@21pravi](https://github.com/21pravi)
