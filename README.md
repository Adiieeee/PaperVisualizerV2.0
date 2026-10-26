# GraphRAG Paper Explorer

Turn a folder of research-paper PDFs into an **interactive knowledge graph** you can explore visually and **ask
questions about in natural language**. Answers are grounded in the papers: the system retrieves facts from the graph
and passages from the text before an LLM writes the answer, and it says so when the papers do not contain the answer.

It runs locally on your machine and is built for a **$0 budget**: free LLM tiers (Google Gemini, optional Groq
fallback) and a local embedding model. No paid or cloud services are required.

---

## Contents

- [What it does](#what-it-does)
- [Who it is for](#who-it-is-for)
- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [The web app](#the-web-app)
- [Managing paper corpora](#managing-paper-corpora)
- [Evaluation and tests](#evaluation-and-tests)
- [Project structure](#project-structure)
- [Security and privacy](#security-and-privacy)
- [Limitations](#limitations)
- [Troubleshooting](#troubleshooting)

---

## What it does

1. **Builds a knowledge graph from PDFs.** An LLM reads every chunk of every paper and extracts entities (papers,
   authors, institutions, methods, datasets, metrics, findings, limitations, tasks) and the relations between them
   (`uses_dataset`, `proposes_method`, `compares`, `has_limitation`, ...). Duplicate entities across papers are merged
   ("BN" and "Batch Normalization" become one node; "ResNet-44" and "ResNet-56" stay separate).
2. **Indexes the text for semantic search.** Paper chunks and summaries of topic clusters ("communities") are embedded
   locally and stored in ChromaDB.
3. **Answers questions with GraphRAG.** Each question is routed to the retrieval strategy that fits it, and the answer
   is generated only from what was retrieved:

   | Question type | Example | How it is answered |
   |---|---|---|
   | About a specific entity | *"Which datasets does SycEval use?"* | Find the node, walk the graph 2 hops, rank the facts by relevance |
   | Thematic | *"What mitigation strategies do these papers propose?"* | Vector search over community summaries |
   | Counting / ranking | *"Which model is used in the most papers?"* | Graph statistics, no vector search |
   | Structural | *"What connects these papers?"* | Entities shared by two or more papers |
   | Off-topic | *"How do I make iced coffee?"* | Politely refused, no retrieval, no extra LLM call |

4. **Lets you explore the graph visually.** A browser page with search, type filters, relation filters, network
   insights, shortest-path discovery between any two nodes, a node detail panel, and a built-in chat box.

Answers are written in the language of the question (for example English or Indonesian). The current corpus is about
**sycophancy in large language models** (8 papers), but nothing is hard-coded to that topic: put different PDFs in
`data/raw/` and rebuild.

## Who it is for

- **Researchers and students** doing a literature review who want to see how papers, methods and datasets relate,
  and ask cross-paper questions ("which papers use hh-rlhf?", "what limitations do they report?").
- **Developers learning GraphRAG**: a complete, readable pipeline (extraction, deduplication, graph construction,
  community detection, hybrid retrieval, reranking, guardrails, evaluation) that runs on a laptop for free.

---

## How it works

The project has two pipelines that share `data/processed/` as their interface.

### Build pipeline (`python main.py build`)

```
PDFs (data/raw/)
  │  1. Ingest: PyMuPDF text, fix ligatures, cut at References/Bibliography, 2000-char chunks (200 overlap)
  ▼
Chunks ──────────────────────────────────────────────────────────────┐
  │  2. Paper metadata: 1 LLM call per PDF (title + popular names)   │
  │  3. Extraction: 1 LLM call per chunk, structured JSON output     │
  │     (retry once, salvage valid items, archived per chunk)        │
  │  4. Clean: drop generic terms, split author lists                │
  │  5. Dedup: exact → normalized → embedding similarity → LLM check │
  │  6. Graph: anchor every entity to its source Paper, link model   │
  │     families (variant_of), reclassify cited papers               │
  │  7. Communities: Louvain over chunk co-occurrence, 1 LLM summary │
  ▼                                                                  ▼
graph.graphml          community summaries ──► ChromaDB ◄── chunk embeddings (local bge-m3)
  │
  └─► manifest.json (integrity), fact embedding cache, static HTML snapshot
```

Extraction is the expensive step. Its results are archived in `data/processed/extraction_cache.json`, so later
rebuilds reuse them unless the chunk text or the extraction prompt/schema changes. A build interrupted by the daily
quota resumes from its checkpoints when you run it again.

### Query pipeline (chat box or `python main.py query`)

```
Question
  │  Router (1 LLM call): entity | thematic | counting | structural | off_topic
  ▼
Retrieval (local, no LLM)
  ├─ entity     → resolve node(s) → 2-hop traversal → rerank facts by similarity to the question
  ├─ thematic   → community summaries above a relevance threshold
  ├─ counting   → rank entities by number of papers, then connections
  ├─ structural → bridge entities linked to ≥ 2 papers
  └─ no node found → chunk search above a relevance threshold, each passage labelled with its paper
  ▼
Answer (1 LLM call): context is wrapped as untrusted DATA; the model answers only from it
```

A normal question costs **2 LLM calls**. Off-topic questions and questions with no relevant passage cost **1** (the
router) and return a fixed message.

### Tech stack

| Layer | Technology |
|---|---|
| PDF parsing | PyMuPDF |
| LLM access | LiteLLM → Gemini (`gemini/gemini-flash-lite-latest` by default), optional Groq fallback |
| Structured output | Pydantic schemas |
| Embeddings | `BAAI/bge-m3` via sentence-transformers (local, multilingual, 1024 dims, CPU or GPU) |
| Graph | NetworkX (GraphML on disk), Louvain community detection |
| Vector store | ChromaDB (local, persistent, telemetry off) |
| Web app | FastAPI + Uvicorn, pyvis / vis.js, vanilla JS panel |

---

## Requirements

- **Python 3.14** (tested with 3.14.6 on Windows 11; other recent versions and OSes should work but are untested).
- A **Google AI Studio API key** (free, no credit card): <https://aistudio.google.com/apikey>.
- Optional: a **Groq API key** (free) for the fallback: <https://console.groq.com/keys>.
- About **2.5 GB of disk** for the embedding model (`BAAI/bge-m3`), downloaded on first run.
- Optional: an NVIDIA GPU with ~3 GB VRAM makes embedding about 5× faster. CPU works fine.

---

## Installation

```bash
git clone https://github.com/Adiieeee/PaperVisualizerV2.0.git
cd PaperVisualizerV2.0

python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt
```

For running the tests, install the dev extras instead: `pip install -r requirements/dev.txt`.
`requirements/lock.txt` lists the exact versions that were tested; it is a reference, not the install path.

**GPU (optional).** Replace the CPU build of PyTorch with a CUDA build, for example:

```bash
pip install torch==2.13.0+cu126 --index-url https://download.pytorch.org/whl/cu126
```

Vectors are stored in fp32, so a store built on GPU can be queried on CPU and vice versa.

---

## Configuration

Copy the template and fill in your key:

```bash
cp .env.example .env      # Windows: copy .env.example .env
```

| Variable | Required | Default | Meaning |
|---|---|---|---|
| `GEMINI_API_KEY` | yes | — | Google AI Studio key |
| `GEMINI_MODEL` | no | `gemini/gemini-flash-lite-latest` | Any LiteLLM `provider/model` name |
| `GROQ_API_KEY` | no | empty | Enables the Groq fallback together with `GROQ_MODEL` |
| `GROQ_MODEL` | no | empty | e.g. `groq/openai/gpt-oss-120b`; leave either Groq variable empty to disable the fallback |
| `EMBEDDING_DEVICE` | no | `auto` | `auto` (GPU if available), `cpu` or `cuda` |

**About the fallback.** Groq is used only when Gemini reports that its **daily** quota is exhausted, and only for
extraction, summaries, routing and answers. Deduplication never falls back, so merge decisions always come from one
model. Per-minute rate limits and server errors are retried on Gemini.

Other tunables (chunk size, relevance thresholds, context budgets, request limits) live in `src/config.py` with
comments explaining how each value was chosen.

---

## Usage

### 1. Add papers

Put your PDF files in `data/raw/`. PDFs over 500 pages or 50 MB are skipped.

### 2. Build the graph

```bash
python main.py build
```

This uses LLM quota: one call per PDF, one per chunk (occasionally two), a few for deduplication, and one per
community. Every call is followed by a 6-second pause to stay under the free-tier rate limit (about 10 requests per
minute), so a corpus of a few papers takes tens of minutes. If the daily quota runs out, the build stops cleanly;
run the same command again after the quota resets and it continues where it stopped.

`python main.py build --reextract` ignores the extraction archive and extracts every chunk again.

### 3. Start the web app

```bash
python server.py
```

It loads the graph and models once, then opens <http://127.0.0.1:8000>. Restart it after every rebuild or corpus
restore.

### 4. Or ask from the command line

```bash
python main.py query -q "Which datasets does SycEval use?"
```

On Windows, if you see encoding errors, run Python in UTF-8 mode: `set PYTHONUTF8=1` (cmd) or
`$env:PYTHONUTF8=1` (PowerShell).

---

## The web app

The page has a search bar (top-left), a filter panel (top-right), a node detail panel (left) and a chat box
(bottom-left):

| Area | What you can do |
|---|---|
| **Search** (top-left) | Type part of a node name, pick a result with the mouse or arrow keys + Enter; the graph zooms to it and opens its details |
| **Entity Types** | Click a type (Paper, Method, Dataset, ...) to highlight all its nodes; hover `?` for what each type means |
| **Network Insights** | Top 5 methods, datasets, metrics and authors by connections, plus the largest model families (e.g. all Llama variants) |
| **Edge Filters** | Show only one relation type (e.g. `uses_dataset`), with an explanation of what it means |
| **Path Discovery** | Pick two nodes and see the shortest chain of relations connecting them |
| **Node details** | Description, the papers it appears in, and its neighbors grouped by type; click any tag to jump to that node |
| **Chat** (bottom-left) | Ask questions in natural language; answers are grounded in the graph and the papers |

Click an empty part of the canvas to clear any highlight. Double-click a node to select it.

The build also writes a standalone snapshot, `data/processed/graph_visualization.html`, which opens without the server
(everything works except the chat). The full UI specification is in [`docs/DESIGN_REFERENCE.md`](docs/DESIGN_REFERENCE.md).

---

## Managing paper corpora

`corpus.py` lets you keep several paper collections and switch between them without rebuilding (and without spending
quota again):

```bash
python corpus.py list                 # list saved corpora
python corpus.py backup sycophancy    # save the active corpus to backups/sycophancy/
python corpus.py clear                # empty the active corpus (asks for confirmation)
python corpus.py restore sycophancy   # bring a saved corpus back
python corpus.py reset-index          # delete build outputs but keep PDFs, metadata and the extraction archive
python corpus.py manifest             # (re)write the integrity manifest for the active corpus
python corpus.py audit                # read-only scan for injection-like text in the graph and chunks
```

Typical flow for a new topic: `backup` the current corpus, `clear`, put new PDFs in `data/raw/`, `python main.py build`.

Note: `build` adds chunks to ChromaDB but never deletes old ones. Use `clear` (after a `backup`) before building a
different corpus, or chunks from the previous papers will still be searchable.

---

## Evaluation and tests

### Retrieval and end-to-end evaluation

`eval/eval_cases.json` holds 62 hand-written questions (entity facts, relations, counting, themes, unanswerable and
adversarial questions, prompt-injection attempts) with the keywords a correct answer must contain.

```bash
python evaluate.py                                   # retrieval quality, free (no LLM calls)
python evaluate.py --e2e                             # full pipeline incl. LLM answers (uses quota)
python evaluate.py --e2e --only id1,id2 --repeat 3   # selected cases, 3 runs each (LLM output varies)
python evaluate.py --budget 80                       # experiment with a different context budget
```

Current results on the 8-paper corpus:

| Metric | Score |
|---|---|
| Retrieval (right facts retrieved) | 38/41 = 93% |
| Guardrail (out-of-domain questions refused without an LLM call) | 5/5 |
| End-to-end (final answer correct, single run) | 52/61 = 85% |

The remaining retrieval failures are documented limitations (see below). Gemini is not fully deterministic, so
compare end-to-end changes with `--repeat` rather than one run.

### Unit and security tests

```bash
pip install -r requirements/dev.txt
python -m pytest tests -q
```

The suite (about 280 tests, ~30 s) needs no network and no LLM quota. It covers prompt sanitization, request guards,
XSS escaping in the frontend, artifact integrity, extraction salvage, quota handling and retrieval logic.
`tests/test_frontend_browser.py` runs in headless Chrome/Edge and is skipped if neither is installed (set
`CHROME_PATH` to point to one).

### Calibration scripts (no LLM quota)

```bash
python -m scripts.calibrate_embedding --model <hf id> --revision <sha>   # thresholds tied to the embedding model
python -m scripts.calibrate_retrieval                                    # relevance thresholds of the vector paths
python -m scripts.community_stats                                        # community size distribution
python -m scripts.test_llm_client                                        # live smoke test of Gemini/Groq (~6 requests)
```

---

## Project structure

```
├── main.py                 CLI: build pipeline and one-off queries
├── server.py               Web app (FastAPI): visualizer at /, chat at POST /api/chat
├── corpus.py               Backup / restore / clear / audit paper corpora
├── evaluate.py             Retrieval and end-to-end evaluation harness
├── src/
│   ├── config.py           All settings, thresholds, entity/relation types, LLM schemas
│   ├── engine.py           Query pipeline (routing → retrieval → answer)
│   ├── ingestion/          PDF reading, section filtering, chunking
│   ├── extraction/         Post-extraction cleaning and hybrid deduplication
│   ├── graph/              Graph builder, analytics/communities, HTML visualizer
│   ├── query/              Router, entity resolver, traversal, fact reranker
│   ├── llm/                LiteLLM client, task prompts, embeddings
│   ├── vectorstore/        ChromaDB wrapper
│   └── security/           Prompt sanitization, request guards, artifact integrity
├── scripts/                Calibration and diagnostic tools
├── tests/                  Unit and security tests (offline)
├── eval/eval_cases.json    Evaluation questions and expected keywords
├── docs/                   UI design reference (and local-only security notes)
├── data/raw/               Your PDFs (not committed)
└── data/processed/         Build outputs (not committed)
```

Build outputs in `data/processed/`:

| File | Content |
|---|---|
| `graph.graphml` | The knowledge graph |
| `chroma_db/` | Chunk and community-summary vectors |
| `paper_metadata.json` | Title and popular model names per PDF |
| `extraction_cache.json` | Archived LLM extraction per chunk (reused across rebuilds) |
| `fact_embeddings.npz` | Cached embeddings of graph facts for reranking (safe to delete) |
| `manifest.json` | Integrity manifest checked at startup |
| `graph_visualization.html` | Standalone visualizer snapshot |
| `similarity_debug.csv` | Every deduplication comparison, for diagnostics |

---

## Security and privacy

This is a **personal tool meant to run on your own machine**, not a public service.

- The server listens on `127.0.0.1` only, accepts only `localhost`/`127.0.0.1` as `Host`, and guards the chat endpoint
  (JSON-only, origin check, body size limit, per-IP rate limit, concurrency limit). There is no login: do not expose it
  to the internet.
- PDF content is treated as untrusted. Everything from the papers or the user enters the prompt inside tags marked as
  DATA, with delimiter tags neutralized, in a single prompt-assembly function. This lowers the risk of prompt
  injection but cannot rule it out, so do not index PDFs from sources you do not trust.
- The page escapes all graph-derived text and LLM output and is served with a hash-based Content Security Policy.
- **Your data leaves your machine**: PDF text, summaries and questions are sent to Google Gemini (and to Groq if the
  fallback is enabled and triggered). Check the providers' free-tier data terms before indexing sensitive or
  copyrighted documents. Embeddings are computed locally.
- Never commit `.env`. If a key ever lands in git history, rotate it.
- Building stays CLI-only; there is no HTTP endpoint that can trigger a build.

---

## Limitations

- **Text only.** Tables are read as text and charts are not read at all, so results that appear only in figures are
  missing from the graph.
- **Appendices are dropped.** Everything after the References heading is cut to avoid citation noise.
- **Two hops.** Facts three relations away (e.g. paper → author → institution → something else) are not retrieved.
- **Paraphrases in other languages** sometimes fail to resolve to the right entity; naming the entity explicitly helps.
- **Free-tier limits.** Daily quotas cap how many papers you can build and how many questions you can ask per day.
- **Single user.** Rate limiting is in-memory and per process; the app is not designed for concurrent public use.
- **LLM variability.** Extraction and answers can differ slightly between runs, even at temperature 0.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `UnicodeEncodeError` / garbled output on Windows | Run with `PYTHONUTF8=1` |
| "Daily quota exhausted" during build | Wait for the quota to reset and run `python main.py build` again; it resumes |
| Chat says the quota is exhausted | Same daily quota as the build; try again after the reset, or configure the Groq fallback |
| Server refuses to start: manifest mismatch | The graph, vector store or embedding model changed. Rebuild, or `python corpus.py manifest` if the change was intentional |
| Build refuses to start: vector dimension mismatch | ChromaDB holds vectors from another embedding model. Run `python corpus.py reset-index` and rebuild |
| Answers mention papers you removed | Old chunks are still in ChromaDB: `python corpus.py clear` (after `backup`) and rebuild |
| Chat in the snapshot HTML says it cannot reach the server | Expected: the chat only works through `python server.py` |
| Changes to the graph do not show up | Restart `server.py`; it loads the graph once at startup |
