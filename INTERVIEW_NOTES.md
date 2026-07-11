# Interview notes

Answers to the questions this project invites. Written for me, published because
nothing here is worth hiding.

## "Walk me through the project."

A retrieval-augmented question-answering system with two answer engines behind one
retrieval path. Documents are chunked and indexed with BM25. A query retrieves ranked
chunks; those exact chunks then feed two independent answering strategies — extractive
selection and LLM generation — shown side by side. The retrieval trace, with cosine-free
BM25 scores and score bars, is the primary UI element.

The point is pedagogical: sharing retrieval means any difference between the two answers
is caused by the answering step alone.

## "Is this GenAI?"

The online engine is. It calls an LLM, which samples tokens from a learned distribution
and produces text that exists nowhere in the source.

The offline engine is not, and I would not describe it as such. BM25 is a ranking
function from classical information retrieval. Extractive QA selects existing sentences.
The image analysis measures pixels. Nothing is generated. Saying otherwise would be an
overclaim, and it is the kind of overclaim that makes an interviewer discount everything
else on the page.

## "Why BM25 instead of embeddings?"

Three reasons.

Correctness: BM25 is the standard lexical baseline dense retrievers are benchmarked
against, and it is strong on short keyword-bearing queries.

Cost: it needs no API, no model, no GPU. That is what allows the hosted demo to run free
and forever as a static page with no backend.

Honesty about the trade-off: BM25 is lexical, so it misses paraphrases with no shared
vocabulary. A dense embedder exists in `providers.py` but is deliberately *not* wired into
the engine yet — I would rather ship one retrieval path that both front-ends and both
answer engines demonstrably share than two paths that quietly diverge.

## "Why is the offline answer worth having if the LLM answer reads better?"

Because it cannot hallucinate. Not "rarely" — there is no generative step in which to
hallucinate. Every sentence is copied unchanged from a retrieved chunk, and a test
asserts each returned sentence is a substring of the source text.

For HR policy, compliance, and legal question-answering, "quotes the handbook, or says it
does not know" is often the required behaviour, and fluency is a secondary concern. The
LLM path exists so the trade-off can be seen rather than asserted.

## "What broke, and how did you find it?"

Four defects, each caught by running the thing rather than reading it.

**Padded citations.** Retrieval returned the top 3 chunks unconditionally, so "How do I
reset my VPN password?" listed `company_policies.md` — score `0.0000` — as a source. Fixed
with a relevance floor; off-topic queries now return nothing and skip the model entirely.
Fixing it broke two tests that had hard-coded "always 3 snippets", which was the correct
outcome: the tests encoded the bug.

**Cached failures.** A transient online error was written to the cache, so a single
network blip poisoned that query permanently. The cache write is now gated on the absence
of an error, and a test proves the retry succeeds once the fault clears.

**Threading.** `sqlite3.connect()` defaults to `check_same_thread=True`. Streamlit reruns
the script in a new thread on every interaction, so the app died on the first click.
Relaxing it is safe here only because CPython's sqlite3 is built serialised
(`sqlite3.threadsafety == 3`) — I checked rather than assumed.

**Dead network code in an "offline" file.** The static build inlined three unreachable
`fetch` calls from the server adapter. The build script now asserts its own output
contains exactly one network call site, and fails otherwise.

**Two retrieval engines behind one project.** The Telegram bot still ran the original
OpenAI-embedding vector store while the web UI had moved to BM25. Same knowledge base,
different pipelines, incompatible SQLite schemas — and nothing failed, because nothing
compared them. Both front-ends now construct the same `engine.Engine`; the legacy store
was deleted rather than left dormant, and a test walks every module asserting no second
`VectorStore` reappears. The migration also surfaced a display bug: the bot rendered
`int(score * 100)` as a relevance percentage, which is correct for cosine but reports an
unbounded BM25 score of 7.57 as "757%".

## "Do the Telegram bot and the web demo run the same code?"

Now, yes — both build `engine.Engine`, so retrieval, the extractive answer, the classical-CV
image analysis and the caching all come from one implementation. Earlier they did not, and
that is the more interesting answer: the bot kept the original embedding-based store after
the web UI moved to BM25. Nothing broke, because nothing tested them together. There is now
a test asserting both files construct `Engine`, and another that walks every module in the
repo to make sure a second retrieval engine has not crept back in.

## "How do you know the browser version behaves like the Python version?"

I do not trust that it does; I test it. The offline engine exists twice by necessity —
the static demo has no backend. `tests/make_expected.py` regenerates reference outputs
from the Python implementation, and the Node suite asserts the JavaScript agrees: BM25
scores to four decimal places, extractive answers character-for-character, chunk counts
per document, and classical-CV metrics against synthetic images with known ground truth.

It also asserts that `ask(query, "offline")` issues zero `fetch` calls. A claim of
"runs offline" that is not tested is a claim, not a property.

## "What would you do next?"

Hybrid retrieval — BM25 for lexical precision, dense embeddings for paraphrase recall,
fused with reciprocal rank fusion. That is the standard next step and it addresses the
known weakness honestly.

Then a real evaluation set. Five synthetic documents cannot support a retrieval
benchmark, so none is reported. With a few hundred labelled query–passage pairs I could
report recall@k and MRR and compare BM25 against a dense retriever on this corpus instead
of relying on the literature.

## "What are you least happy with?"

Chunking is a fixed 500-character window. Sentence- and heading-aware chunking would
retrieve better. I kept the naive version because identical chunking in Python and
JavaScript is what makes the parity test meaningful, and I judged the demonstrable
correctness guarantee more valuable than a marginal retrieval gain. That is a defensible
trade, but it is a trade.

The browser-direct Claude call in the static build is the one path with no automated
test, because it depends on a CORS policy I cannot exercise in CI.
