# Contributing

## Setup

```bash
pip install -r requirements-dev.txt
```

## Before opening a pull request

```bash
python tests/make_expected.py     # regenerate parity fixtures
pytest tests/ -q                  # 93 tests
node tests/test_static_parity.mjs # 45 tests
python build_static.py            # docs/index.html must be regenerated and committed
```

CI runs exactly these four commands and additionally fails if `docs/index.html` is
out of date with respect to `static/` or `knowledge_base/`.

## The one rule

**If you change the offline engine, change it twice.**

`bm25.py`, `extractive.py` and `imageanalysis.py` are mirrored in JavaScript inside
`build_static.py`, because the static demo has no backend. The parity suite compares them
directly — BM25 scores to four decimal places, extractive answers character-for-character —
so a change to one without the other will fail CI, loudly and immediately.

That duplication is a deliberate cost. It buys a demo that runs on GitHub Pages with no
server, no API key and no expiry.

## Things that will be rejected

- Network calls in the offline engine. The build asserts exactly one `fetch` call site
  exists in `docs/index.html`, and it belongs to the online engine.
- `localStorage` or any browser storage in the UI.
- Describing the offline engine as generative AI. It ranks and selects; it does not generate.
- A relaxed relevance threshold that lets zero-score chunks be cited as sources.
- A second retrieval engine. `server.py` and `bot.py` must both construct `engine.Engine`;
  a test walks every module to enforce it. Front-ends own presentation, nothing else.

## Adding knowledge-base documents

Drop a `.md` or `.txt` file into `knowledge_base/`, then regenerate:

```bash
python tests/make_expected.py && python build_static.py
```

Both the parity fixtures and the inlined corpus in `docs/index.html` are derived from
that directory, so they must be rebuilt together.
