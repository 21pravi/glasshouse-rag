# Methodology

## Retrieval: BM25 Okapi

Each chunk is scored against the query as

$$\text{score}(D,Q)=\sum_{t \in Q} \text{IDF}(t)\cdot\frac{f(t,D)\,(k_1+1)}{f(t,D)+k_1\left(1-b+b\,\frac{|D|}{\text{avgdl}}\right)}$$

with $\text{IDF}(t)=\ln\!\left(1+\frac{N-n(t)+0.5}{n(t)+0.5}\right)$, $k_1=1.5$, $b=0.75$.

Two properties are why BM25 rather than raw TF-IDF, and both are asserted by tests
rather than assumed:

**Term-frequency saturation.** The $\frac{f(k_1+1)}{f+k_1(\cdot)}$ form is bounded above
by $k_1+1$. A term occurring fifty times is *not* fifty times more relevant than one
occurring once. `test_bm25_saturates_term_frequency` builds a document repeating a term
50× and asserts it scores higher than a single occurrence but less than 4× higher.

**Document-length normalisation.** The $b\,|D|/\text{avgdl}$ term penalises long
documents that accumulate matches by sheer size. `test_bm25_penalises_long_documents`
pads a document with 60 repetitions of filler and asserts the short one still wins.

### Why lexical retrieval at all

Dense embeddings are the fashionable choice, but BM25 is the baseline they are measured
against, and it is competitive on short, keyword-bearing queries of exactly the kind a
knowledge-base assistant receives. It also requires no API, no model download, no
GPU, and no index rebuild when a key rotates — which is what makes a permanently free,
zero-backend demo possible at all.

The trade-off is real: BM25 is lexical, so a query sharing no vocabulary with the source
("time off" vs "vacation leave") will miss.

`providers.OpenAIEmbedder` implements a dense embedder, but **`Engine` does not currently
use it** — retrieval is BM25 unconditionally, in both the web UI and the Telegram bot, so
that both front-ends and both answer engines see identical chunks. Wiring dense retrieval
in as a selectable strategy, and fusing it with BM25 via reciprocal rank fusion, is the
next piece of work. Until it is done, this document does not claim it exists.

### Thresholding

Chunks scoring below `0.1` are discarded rather than padded to `top_k`. An earlier
revision returned the top 3 unconditionally, so a question about VPN passwords cited
`company_policies.md` at score `0.0000` as a source. Citing a document that shares no
terms with the query is worse than citing nothing. If no chunk clears the threshold the
engine returns "Nothing in the knowledge base is relevant" **without calling a model**.

## Chunking

500 characters, 100 overlap, split on character count.

Naive, and deliberately so. Sentence- or heading-aware chunking would perform better,
but chunking is not what this project is demonstrating, and a character window keeps the
Python and JavaScript implementations trivially identical — which is what the parity
suite depends on. The 100-character overlap prevents an answer that straddles a boundary
from being lost.

## Answering, offline: extractive QA

1. Split the retrieved chunks into sentences. Markdown headings are dropped; fragments
   under 25 characters are dropped.
2. Build a *second* BM25 index over the sentence pool.
3. Score every sentence against the query.
4. Nudge sentences containing cue words matching the question type (`how` → "visit",
   "select", "submit"; `when` → "within", "before") by 1.15×. A mild prior, not a rewrite.
5. Take the top two distinct sentences, restore their original reading order, and return
   them unchanged with a source attribution.

Nothing is generated. Every character of the answer appears in a source document, which
`test_extractive_answer_is_verbatim_from_source` checks by asserting each returned
sentence is a substring of the retrieved text.

**This is the project's central claim.** An extractive system cannot hallucinate, because
there is no generative step in which to hallucinate. It can be *wrong* — it may select an
irrelevant sentence — but it cannot invent a policy that does not exist. For compliance,
HR, and policy Q&A that failure mode is strictly preferable.

The cost is fluency. Extracted sentences do not flow, do not synthesise across sources,
and cannot answer "why" questions that require reasoning over several passages. The
online engine is included precisely so the trade-off is visible rather than argued.

## Answering, online: grounded generation

The same ranked chunks are formatted into a context block, each tagged with its source,
and passed to an LLM with instructions to answer only from the context and to name the
document relied upon. Temperature `0.3`.

This is genuine generative AI: the model samples tokens from a learned distribution, so
its output does not exist verbatim in any source. That is why it reads well, and why it
can drift.

## Vision, offline: classical computer vision

No model. The image is downscaled to 256 px on its long edge, then measured:

| Property | Method |
|---|---|
| Dominant palette | median-cut quantisation to 5 colours, nearest-neighbour naming in RGB |
| Brightness | mean luminance, $0.299R + 0.587G + 0.114B$ |
| Contrast | RMS of the luminance plane |
| Edge density | fraction of pixels whose gradient magnitude exceeds 40 |
| Sharpness | variance of the edge response |
| Colourfulness | Hasler & Süsstrunk (2003), $\sigma_{rgyb} + 0.3\,\mu_{rgyb}$ |
| Greyscale | colourfulness < 8 **and** mean \|R − B\| < 6 |

Verified against synthetic images with known ground truth: a solid navy fill must be
named `navy` with near-zero edge density; uniform white must register as greyscale yet
still be captioned "white tones"; uniform noise must exceed 0.2 edge density.

The output states in its own text that it is not a vision model and identifies no
objects or people. Claiming otherwise would be the image-analysis equivalent of a
hallucination.

## What is *not* claimed

- The offline engine is not GenAI. Ranking functions and sentence selection are classical
  information retrieval, not generation.
- The knowledge base is five short synthetic documents (12 chunks). Retrieval quality on
  a corpus of that size says nothing about behaviour at scale.
- No retrieval benchmark (MRR, nDCG, recall@k) is reported, because five documents cannot
  support one. The tests verify *correctness of implementation*, not competitive quality.
- Retrieval is BM25 only. The dense embedder in `providers.py` is not wired into `Engine`.
