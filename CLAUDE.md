# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

GraphRAG Paper Explorer: builds a knowledge graph (NetworkX) from PDFs in `data/raw/`, then answers questions over it by combining graph traversal, ChromaDB vector search, and LLM generation (Gemini via LiteLLM, optional Groq fallback). The app is an interactive graph visualizer (pyvis/vis.js) with a chatbot panel, served by FastAPI. Docs, UI text and most prompts are in Indonesian; answers are generated in the language of the question.

**Hard constraint: $0 budget.** Free tiers without a credit card only: Gemini via Google AI Studio (`gemini/gemini-flash-lite-latest`) as primary, Groq (e.g. `groq/openai/gpt-oss-120b`) as optional fallback, both through LiteLLM (no proxy, no cache). Embeddings run locally (`BAAI/bge-m3`, multilingual, 1024 dims; GPU when available, else CPU), never through LiteLLM. No paid/cloud services.

## Commands

Setup: `pip install -r requirements.txt` (`requirements-dev.txt` adds pytest/httpx), and put `GEMINI_API_KEY=...` in `.env` (copy `.env.example`; loaded in `src/config.py`). Optional: `GROQ_API_KEY` + `GROQ_MODEL` enable the Groq fallback (either empty = no fallback); `GEMINI_MODEL` overrides the Gemini model (LiteLLM `provider/model` names). `requirements.lock.txt` is a reproducibility reference (Python 3.14.6, Windows), not the install path.

GPU (optional, ~5x faster embedding): `pip install torch==2.13.0+cu126 --index-url https://download.pytorch.org/whl/cu126` replaces the CPU torch wheel (tested on a GTX 1650 4 GB; bge-m3 fp32 peaks at ~2.6 GB VRAM). `EMBEDDING_DEVICE=auto|cpu|cuda` (default `auto`). Vectors stay fp32, so a store built on GPU is queried correctly on CPU (cosine 1.0). fp16 was slower on that card and shifts vectors, so it is not used.

```powershell
python main.py build                    # PDFs in data/raw/ -> graph.graphml + chroma_db + manifest.json (uses LLM quota)
python main.py query -q "pertanyaan"    # CLI query
python server.py                        # web app at http://127.0.0.1:8000 (visualizer + POST /api/chat)
python evaluate.py                      # retrieval-level eval, free (no LLM), reads eval/eval_cases.json
python evaluate.py --e2e                # full engine.query per case (router + generation, uses quota)
python -m pytest tests -q               # unit/security tests: no network, no LLM quota (~10 s)
python corpus.py list|backup <name>|clear|restore <name>   # swap paper corpora without rebuilding
python corpus.py reset-index            # delete build outputs (chroma_db, graph, manifest, checkpoints); keeps PDFs + paper_metadata.json
python corpus.py manifest               # write manifest.json for the ACTIVE corpus (no rebuild)
python corpus.py audit                  # read-only scan of graph + chunks for injection-like text
python -m scripts.test_llm_client       # live LiteLLM smoke test: Gemini, simulated-quota Groq fallback, schemas (~6 requests)
python -m scripts.calibrate_embedding --model <hf id> --revision <sha>   # recalibrate embedding thresholds, no LLM quota
```

- `evaluate.py` is the retrieval regression harness; run it after any query-time change and compare the score before → after. It has no per-case filter. Cases with `e2e_only` / `expected_refuse` are excluded from the retrieval score and only graded by `--e2e` (`must_not_contain` flags leaked/obeyed injected content).
- `pytest` covers sanitization, prompt assembly, request guards, frontend XSS, artifact integrity. Run a single file with `python -m pytest tests/test_guards.py -q`; `tests/test_engine_blocks.py` imports torch (~9 s) and `tests/test_frontend_browser.py` needs Chrome/Edge (auto-skipped otherwise; set `CHROME_PATH`). No linter/formatter.
- On Windows, run Python with `PYTHONUTF8=1` when output contains non-ASCII (`evaluate.py` prints `↳`; cp1252 consoles crash).

## Architecture

Two pipelines share `data/processed/` (gitignored) as their interface: `graph.graphml`, `chroma_db/`, `paper_metadata.json`.

### Build (`main.py: build_pipeline`, everything is orchestrated inline here)

1. **Ingest** (`src/ingestion/document.py`): PyMuPDF → fix ligatures → truncate at the first `References/Bibliography/Acknowledgements` heading located past 20% of the doc (`SECTION_FILTER_*` in `config.py`; this also drops any appendix after it — a deliberate choice) → 2000-char chunks / 200 overlap. `chunk_id = "{pdf filename}_chunk_{n}"`; doc id is recovered from it via `rsplit("_chunk_")`.
2. **Paper metadata**: one LLM call per PDF → title + popular model names (`paper_metadata.json`). Used to inject the title into extraction prompts and to link model nodes (e.g. "VGG") to their Paper node, since papers rarely name themselves that way. Also auto-extends `KNOWN_ACRONYMS` from "Full Name (ACR)" patterns in the corpus.
3. **Extract**: `LLMService.extract_entities` per chunk with structured output (Pydantic schemas in `config.py`), checkpointed after every chunk.
4. **Clean** (`extraction/cleaner.py`): generic-term blocklist, min name length, author splitting/"et al." stripping. Runs before dedup.
5. **Dedup** (`extraction/deduplicator.py`, `HybridDeduplicator`): per-type namespace; exact → normalized (case/punct/acronym) → embedding cosine ≥ `DEDUP_SIMILARITY_THRESHOLD` (0.85) → LLM confirm. Prompt guards: numbers/versions never merge (ResNet-44 ≠ ResNet-56) and no hypernym merging. Every comparison is logged to `similarity_debug.csv`; state is checkpointed to `checkpoint_dedup.json`.
6. **Graph** (`graph/builder.py`): `add_knowledge_with_provenance` anchors every non-Paper node to its source Paper with a synthetic `mentioned_in` edge; then `link_models_to_papers`, `reclassify_noncanonical_papers` (cited papers → Method), `link_entity_families` (`variant_of` edges from version-suffix names, purely regex-based), `merge_family_aliases` (few LLM calls).
7. **Communities** (`graph/analytics.py`): Louvain over the graph **excluding Paper nodes**; each community summarized by the LLM and stored in the `community_summaries` Chroma collection. Chunks go in the `chunks` collection. Then `manifest.json` (integrity manifest) and a static `graph_visualization.html` snapshot are written.

### Query (`src/engine.py: GraphRAGEngine.query`)

`QueryRouter` (one LLM call) classifies intent, then:
- `counting` → `GraphAnalytics.get_counting_stats` (degree ranking, no vector search)
- `structural` → `get_bridge_entities` (non-Paper nodes linked to ≥2 Papers)
- `thematic` → vector search over community summaries
- `entity` → `QueryResolver.resolve_entities` (multi-entity; layers: paper-context model→Paper, exact, partial Paper-title overlap, verbatim node-name-in-query with longest-span-wins, fuzzy >0.8, embedding-nearest-node ≥0.72) → `GraphTraversal.traverse` (2-hop BFS, per-hop relation budget, Paper nodes not expanded except the *source* paper of a non-Paper start node) → `format_traversal_context` (groups hop-1 by relation, hop-2 as `A → rel → B` chains). If nothing resolves, falls back to chunk vector search.

Then `generate_final_answer(query, blocks)` with the context as typed blocks `(kind, text)` — `graph` / `text` / `community` (`src/llm/prompts.py`). Each query = 2 LLM calls (route + generate). The context format and the generation prompt's rules about "RELASI LANGSUNG / TIDAK LANGSUNG" are coupled — change them together (`tests/test_prompt.py` pins both).

### LLM layer (`src/llm/llm_client.py`, `src/llm/llm_service.py`)

Every LLM call goes `LLMService` method → `call_llm(alias=...)` → LiteLLM `Router`. `llm_client.py` is the transport (Router, aliases, fallback); `LLMService` only builds the task prompts, parses Pydantic results, retries transient errors with tenacity, and sleeps 6 s per call; `last_model` holds the model that actually answered. Two aliases:
- `general` (extraction, paper metadata, community summaries, query routing, final answer): Gemini; switches to Groq **only** when Gemini returns a daily-quota 429. Per-minute 429s, 5xx and timeouts are re-raised and retried on Gemini by tenacity. The fallback is done in `call_llm`, deliberately not via Router `fallbacks=` (those switch on every error); the Router has `num_retries=0` and no cooldowns so retries don't stack.
- `dedup` (`resolve_dedup`, `resolve_family_merge`): Gemini only, never falls back — merge decisions must come from one model.

Structured output is sent via `response_format_for(schema)`: a non-strict JSON schema with Optional fields flattened to nullable. Passing the Pydantic class directly makes LiteLLM use OpenAI strict mode (every field required), which measurably changed Gemini's extraction output. The answering model is recorded per chunk (`checkpoint_extraction.json` → `data/processed/extraction_models.json`) and per dedup decision (`checkpoint_dedup.json`, `similarity_debug.csv`).

### App (`server.py`, `src/security/guards.py`, `src/graph/visualizer.py`)

`GraphVisualizer.build_html()` renders pyvis (inline resources; the Bootstrap CDN tags pyvis adds are stripped so the page is fully offline) and injects a large custom panel (`_PANEL_CSS/_PANEL_HTML/_PANEL_JS` string constants + a `GR_DATA` JSON blob) before `</body>`. The panel's chat box calls `POST /api/chat`, which only exists under `server.py` (the static snapshot degrades gracefully). `server.py` exposes `create_app(engine, html)` (importable without torch); the engine and graph are loaded **once at startup** in `main()` — restart `server.py` after a rebuild or `corpus.py restore`. `DESIGN_REFERENCE.md` is the UI spec. Write HTML as UTF-8 yourself (pyvis's `write_html` uses the OS codepage and crashes on Windows).

## Security conventions (threat model: personal, local-only tool; public repo others may fork)

- **One choke point for prompts:** question + context become a prompt only in `build_answer_prompt` (`src/llm/prompts.py`, called by `LLMService.generate_final_answer`). Everything user- or document-derived goes through `neutralize_tags()` and sits inside a tag the prompt declares to be untrusted DATA. New prompt tag ⇒ add it to `PROTECTED_TAGS` in `src/security/sanitize.py`. Never interpolate context into a prompt anywhere else.
- **Neutralize at prompt time, never at ingest:** rewriting chunk text would change ChromaDB embeddings (vector layer must stay intact). `sanitize_entity_name` runs in `cleaner.py` and only takes effect on the next rebuild.
- **Request guard order** (`ChatGuard`, pure ASGI, runs before FastAPI parses the body): body size → `Content-Type: application/json` → Origin → rate limit → body read cap → concurrency. The fixed `Host` allow-list (`config.ALLOWED_HOSTS`) is what stops DNS rebinding — comparing Origin to the request's own Host would not.
- **Frontend:** `renderMarkdown` must escape first, then add whitelisted tags; use `esc()` on every graph-derived value put into `innerHTML`. CSP `script-src` is hash-based (`build_index_csp`) — a new inline `<script>` is fine (hash recomputed), an inline `onclick=` attribute or an external script is not.
- **Build stays CLI-only.** Do not expose build/reindex over HTTP.
- **Second provider:** when the Groq fallback triggers, user questions and paper chunks are sent to Groq too. The prompt-injection defenses were tuned on Gemini; `python evaluate.py --e2e` is the way to re-check them on another model.
- **Invisible/escape characters:** the file-writing tooling used in past sessions decodes `\uXXXX` in content into literal characters (and literal zero-width chars vanish on copy). Build such characters with `chr()` (see `sanitize.py`; `tests/test_sanitize.py` guards it).
- **Artifacts:** `manifest.json` (embedding model + revision, graph SHA-256, counts) is verified at engine start (`check_artifacts`). After a legitimate rebuild it is rewritten by `build`; after changing `EMBEDDING_MODEL_NAME/REVISION` the corpus must be rebuilt. Checkpoints and `graph.graphml` are written atomically (`atomic_write*`).

## Things that are easy to get wrong

- **Graph is an undirected `nx.Graph`** with at most one edge per node pair (the first relation wins on duplicates); `source/target` direction in the context is derived from traversal order, not stored. GraphML round-trips force all attributes to strings.
- **Synthetic relations** `mentioned_in` and `variant_of` are added by the builder, not the LLM (`variant_of` is not in `RELATION_TYPES`). They are excluded from degree counts in both `analytics.get_counting_stats` and `visualizer._insights` — keep the two in sync.
- **Adding an entity/relation type** touches `config.py` Literals, the extraction prompt in `llm_service.py`, and the visualizer's `NODE_COLORS`/`EDGE_COLORS`/`*_DESCRIPTIONS`.
- **Changing the embedding model** (`EMBEDDING_MODEL_NAME/REVISION/DIMENSION`): `DEDUP_SIMILARITY_THRESHOLD` and `RESOLVER_SIM_THRESHOLD` (config.py) are tied to the model's score distribution, so recalibrate them with `scripts/calibrate_embedding.py`. It re-scores the last build's `similarity_debug.csv` and sweeps the resolver on a temporary store; judge the listed LLM merges by hand, because the old LLM approved some wrong merges. Then run `corpus.py reset-index` and rebuild. The build refuses to start (before any LLM call) if Chroma holds vectors of another dimension. `EMBEDDING_MAX_SEQ_LENGTH` (1024, capped at the model's own limit) is part of the manifest too.
- **Vector layer is protected.** Post-extraction cleaning, author normalization, and graph post-processing are graph-only by design; don't change them in ways that alter chunk embeddings, the embedding-based dedup, or the ChromaDB collections without checking the effect on retrieval.
- **Build-time changes are expensive.** The extraction and dedup checkpoints are kept until the build completes and deleted only then, so any change to extraction, dedup, or graph building means re-extracting every chunk with LLM quota. Prefer query-time changes (measurable free with `evaluate.py`) and `corpus.py backup` before `clear`. `chroma_db` is upserted, never wiped by `build` — stale chunks from a previous corpus survive unless you `corpus.py clear`.
- **Quota/rate handling**: every LLM call `time.sleep(6.0)`s afterward (~10 req/min). Daily-quota detection is a message match (`PerDay` / `per day`) in `llm_client.is_daily_quota_error`, only tested against synthetic errors. When no fallback is left, `DailyQuotaExceededError` (= `LLMQuotaError`) is raised and not retried. `build` then stops with a message at whatever stage hit it (extraction, dedup after saving its checkpoint, family merge, community summaries) and re-running resumes from the checkpoints. `graph.graphml` is saved only after the last quota-dependent stage. `server.py` chat and `evaluate.py --e2e` consume the same daily quotas.
- **Dedup resume** (`main.py` + `deduplicator.py`): knowledge is reloaded from the extraction checkpoint with **raw** names, so skipped items get the saved mapping applied (`_apply_saved_mapping`). The resume point is the completed *prefix* (`dedup_resume_index`), and `rebuild_normalized_index()` restores the in-memory step-2 index. `similarity_debug.csv` is still overwritten on every run, including resumes.
- **`import litellm` order**: importing litellm before `src.llm.llm_client` fetches its model cost map from GitHub. `llm_client` sets `LITELLM_LOCAL_MODEL_COST_MAP` before importing; tests also set it in `conftest.py`. Keep that order in new code.
- **Dead code**: `src/extraction/extractor.py` (`KnowledgeExtractor`), `GraphBuilder.add_knowledge`, `load_graph`, and `traverse_1_hop` are unused by `main.py`/`engine.py`. The live build path calls `LLMService.extract_entities` and `add_knowledge_with_provenance` directly. `google-genai` is no longer used or required (everything goes through LiteLLM).
- **Local-only docs**: internal planning/spec/report docs are gitignored (see `.gitignore`) and may be absent in a fresh clone. Where present, most are stale — trust the code over them. Exception: a gitignored plan file the user asks you to work on is authoritative for that task; read it fully first and follow its working rules.

## Workflow conventions

- Don't add new retrieval/resolution heuristics without a measured failing case in `eval/eval_cases.json`; report the `evaluate.py` score before → after.
- Current baseline (2026-07-31 corpus, 8 sycophancy papers, bge-small-en-v1.5): retrieval 29/31 = 94% (remaining failures are documented Indonesian-paraphrase resolution limits). The bge-m3 calibration on that graph gave 30/31; replace this baseline after the bge-m3 rebuild.
