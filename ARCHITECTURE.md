# Architecture

## The central decision

Both *front-ends* — the web UI and the Telegram bot — construct the same `engine.Engine`.
Both *answer engines* inside it share **one retrieval path**. The offline and online answers are computed
from the same ranked chunks, and a test enforces it:

```python
assert off["snippets"] == on["snippets"] == both["snippets"]
```

Had the offline engine used BM25 while the online engine used dense embeddings, a
better online answer would be unattributable — better retrieval, or better generation?
Two variables, no signal. Sharing retrieval reduces the comparison to a single question:
*what does generation add on top of retrieval?*

## Request flow

```
POST /api/ask {query, mode}
        │
        ├── cache lookup, keyed on (mode, normalised query)
        │
        ├── BM25Index.search(query) ──► ranked chunks, score > 0.1
        │                                (empty ⇒ "nothing relevant", no model call)
        │
        ├── mode ∈ {offline, both}: extractive.answer(query, chunks)
        │
        └── mode ∈ {online, both}:  chat.complete(system, context + query)
                                     └── failure is caught, reported per-panel,
                                         and never cached
```

The offline answer is computed first and independently. An online failure — dead key,
network partition, rate limit — leaves the offline card fully populated and puts the
error in the online card. This is tested by injecting `ConnectionError` mid-request.

## Modules

| Module | Responsibility | Dependencies |
|---|---|---|
| `bm25.py` | Okapi BM25 ranking, tokenisation, stopwords | none |
| `extractive.py` | sentence splitting, sentence-level BM25, selection | `bm25` |
| `imageanalysis.py` | palette, exposure, contrast, edges, colourfulness | `numpy`, `pillow` |
| `corpus.py` | document loading and chunking | none |
| `providers.py` | `OpenAI*` / `Anthropic*` / `Offline*` behind one interface | lazy |
| `engine.py` | index, search, cache, both answer paths | `bm25`, `extractive` |
| `server.py` | HTTP surface, validation, error mapping | `fastapi` |
| `bot.py` | Telegram surface over the *same* `Engine` | `python-telegram-bot` |
| `build_static.py` | compiles UI + offline engine into a single HTML file | none |

`openai` and `anthropic` are imported lazily and guarded, so an offline-only install
(`pip install -r requirements.txt`) imports every module cleanly with neither package
present. Attempting to *use* an online provider without its package raises a message
that says which package to install.

## One engine, two front-ends

`server.py` and `bot.py` are thin adapters. Neither contains retrieval, ranking, answering
or image analysis; both call `Engine`. This was not always true — the bot ran an
OpenAI-embedding vector store long after the web UI moved to BM25, and their SQLite schemas
were not even compatible. The legacy store was deleted rather than deprecated, and
`test_no_second_retrieval_engine_survives_anywhere` walks every module in the repo asserting
that no `class VectorStore` and no `generate_rag_answer` reappear.

Front-end adapters own only presentation. `bot.py` renders BM25 scores as raw floats, for
instance, because BM25 is unbounded: the version inherited from the cosine-era code printed
`int(score * 100)` and would have reported a score of 7.57 as "757% relevance".

## Two implementations, one behaviour

The offline engine exists twice: in Python (`bm25.py`, `extractive.py`,
`imageanalysis.py`) and in JavaScript (compiled into `docs/index.html`). Duplication is
deliberate — it is what lets the demo run with no backend — so it is defended by a
parity suite rather than by hope.

`tests/make_expected.py` regenerates reference outputs from the Python implementation.
`tests/test_static_parity.mjs` then asserts the JavaScript agrees:

- chunk counts, overall and per document
- BM25 scores, to four decimal places, across nine queries
- extractive answers, **character-for-character**
- classical-CV metrics on synthetic images with known ground truth
- `ask(query, "offline")` issues zero `fetch` calls; `"both"` issues exactly one

If either implementation drifts, CI fails.

## Caching

Keyed on `md5(mode + "::" + query.lower().strip())`.

Two properties are enforced by tests:

1. **Per-mode keys.** An offline answer must never be served for an online request.
2. **Failures are never cached.** An earlier revision cached the error payload, so one
   network blip poisoned that query forever. The write is now gated on
   `online_error is None`, and a test proves the retry succeeds once the fault clears.

## Error mapping

| Condition | Status | Behaviour |
|---|---|---|
| empty / oversized query | 422 | Pydantic validation |
| unknown mode | 422 | rejected before any work |
| unsupported image MIME | 415 | rejected before reading the body |
| empty upload | 400 | |
| image > 5 MB | 413 | |
| no chunk above threshold | 200 | "Nothing in the knowledge base is relevant" — **no model call** |
| online provider failure | 200 | offline answer intact, error surfaced in the online panel |

Note the last two rows. An off-topic query never reaches the model, so it costs nothing;
and a provider outage degrades the page rather than breaking it.

## The static build

`build_static.py` inlines the stylesheet, the shared `app.js`, the ported offline
engine, and the corpus into one file, then asserts its own output:

```python
assert "ServerAdapter" not in body          # no dead backend code
assert body.count("fetch(") == 1            # exactly one network call site
assert "api.openai.com" not in body
```

That third assertion caught a real defect: three unreachable `ServerAdapter` fetch calls
were being shipped inside a file described as offline-safe. The build now fails rather
than emit them.
