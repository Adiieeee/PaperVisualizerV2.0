import argparse
import hashlib
import os
import json
from collections import Counter

from src.ingestion.document import DocumentProcessor
from src.llm.llm_service import LLMService, DailyQuotaExceededError, build_extraction_prompt
from src.llm.embedding import EmbeddingClient
from src.extraction.cleaner import clean_knowledge_list, preprocess_authors
from src.extraction.deduplicator import HybridDeduplicator, extract_acronyms_from_texts, register_acronyms
from src.graph.builder import GraphBuilder
from src.graph.analytics import GraphAnalytics
from src.vectorstore.chromadb_store import ChromaStore
from src.engine import GraphRAGEngine
from src.query.graph_rerank import GraphFactReranker
from src.config import (
    COMMUNITY_MIN_SIZE, FACT_EMBEDDINGS_PATH, ExtractedKnowledge, LLMExtractedKnowledge, load_paper_metadata, save_paper_metadata,
)
from src.security.integrity import (
    ArtifactIntegrityError, atomic_write_json, build_manifest, chroma_stats, write_manifest,
)

# ──────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────
# The extraction archive = the extraction checkpoint that is NOT deleted after the build: the per-chunk LLM result
# (+ status ok/retried/salvaged/failed, model, text hash). Extraction is the most expensive stage and is not
# deterministic; the archive allows auditing and rebuilding without re-extracting (reset-index
# keeps it). Reused only if the chunk text AND the prompt/schema fingerprint match.
EXTRACTION_CACHE_PATH = "data/processed/extraction_cache.json"
EXTRACTION_CACHE_VERSION = 1


def extraction_fingerprint() -> str:
    """Changes when the extraction prompt or the schema sent to the LLM changes."""
    blob = build_extraction_prompt("", None) + json.dumps(LLMExtractedKnowledge.model_json_schema(), sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_extraction_cache(fingerprint: str) -> dict:
    """{chunk_id: entry} made with the current prompt/schema; {} if there is none yet or it is stale."""
    if not os.path.exists(EXTRACTION_CACHE_PATH):
        return {}
    try:
        with open(EXTRACTION_CACHE_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except ValueError as e:  # corrupt JSON: do NOT silently restart and burn the quota again
        raise ArtifactIntegrityError(
            f"{EXTRACTION_CACHE_PATH} is corrupt ({e}). Fix it by hand, move it aside, or run "
            f"'python main.py build --reextract' to extract every chunk again."
        ) from e
    if payload.get("version") != EXTRACTION_CACHE_VERSION or payload.get("fingerprint") != fingerprint:
        print("[Extract] Arsip ekstraksi dibuat dengan prompt/skema lain: semua chunk diekstrak ulang.")
        return {}
    return payload.get("chunks", {})


def _save_extraction_cache(fingerprint: str, entries: dict):
    # Atomic (temp file + os.replace): this file holds costly LLM output; a crash mid-write must
    # never leave truncated JSON that makes the build impossible to resume.
    atomic_write_json(EXTRACTION_CACHE_PATH, {
        "version": EXTRACTION_CACHE_VERSION, "fingerprint": fingerprint, "chunks": entries,
    }, indent=1)


def _reusable(entry, text: str) -> bool:
    return bool(entry) and entry.get("status") != "failed" and entry.get("text_sha256") == _text_sha256(text)


def _print_extraction_summary(entries: dict):
    by_status, by_model = Counter(), Counter()
    for e in entries.values():
        by_status[e["status"]] += 1
        by_model[e.get("model") or "?"] += 1
    print(f"[Extract] Status: {dict(by_status)} | model: {dict(by_model)}  (detail: {EXTRACTION_CACHE_PATH})")
    salvaged = {cid: e["dropped"] for cid, e in entries.items() if e["status"] == "salvaged"}
    if salvaged:
        print(f"[Extract] Diselamatkan sebagian (item dibuang): {salvaged}")

def _vector_store_mismatch(chroma, embed):
    """Error message if ChromaDB holds vectors of a different dimension than the current embedding model, else None."""
    stored = chroma_stats(chroma)["dimension"]
    if stored is None:  # empty store: safe
        return None
    current = len(embed.embed_text("dimension probe"))
    if stored == current:
        return None
    return (f"ChromaDB berisi vektor {stored} dimensi, model embedding sekarang menghasilkan {current} "
            f"dimensi (model diganti?). Build dihentikan sebelum memakai kuota LLM. Kosongkan indeks "
            f"dulu: `python corpus.py reset-index` (PDF dan paper_metadata.json dipertahankan).")


def _stop_for_quota(stage: str, exc: Exception):
    """Daily quota exhausted after dedup. The extraction archive + dedup checkpoint still exist, so a re-run
    after the quota resets skips both and repeats only the graph/community stages."""
    print(f"\n[Build] Stopped during {stage}: {exc}")
    print("[Checkpoint] Extraction archive + dedup checkpoint kept; re-run 'python main.py build' after the "
          "quota resets (graph, family merge and community summaries are redone).")


# ──────────────────────────────────────────
# Visualization helper
# ──────────────────────────────────────────
def _generate_visualization(graph, output_path: str = "data/processed/graph_visualization.html"):
    """Generate the standalone interactive HTML (portable snapshot; not opened automatically —
    use `python server.py` for the live app). Wrapped so a visualization hiccup never discards
    a completed (quota-costly) build."""
    try:
        from src.graph.visualizer import GraphVisualizer

        GraphVisualizer(graph).generate_html(output_path)
        print(f"[Viz] Saved standalone visualization to {output_path}")
        print("[Viz] Run 'python server.py' for the interactive app + chatbot.")
    except Exception as e:
        print(f"[Viz] Could not generate the visualization: {e}")
        print("[Viz] The graph is still saved; you can generate it manually later.")


# ──────────────────────────────────────────
# Build Pipeline
# ──────────────────────────────────────────
def build_pipeline(reextract: bool = False):
    print("=== Starting Build Pipeline ===")
    
    # 1. Ingestion
    doc_processor = DocumentProcessor()
    chunks = doc_processor.process_directory("data/raw")
    if not chunks:
        print("No chunks extracted. Please place PDFs in data/raw/")
        return
        
    print(f"Total chunks to process: {len(chunks)}")

    # 2. Setup Clients
    llm = LLMService()
    embed = EmbeddingClient()

    # Preflight (before any LLM call): chunks are upserted into the existing Chroma collections only at
    # step 7. A store built by another embedding model (other dimension) would reject them there —
    # after the whole LLM quota of the build has been spent.
    problem = _vector_store_mismatch(ChromaStore(), embed)
    if problem:
        print(f"\n[ERROR] {problem}")
        return

    # 2a. Auto paper metadata (replaces the old hardcoded PAPER_METADATA): one LLM call per PDF
    #     extracts the title + popular model names, cached to JSON (resumable).
    paper_metadata = load_paper_metadata()
    first_chunk_of_doc = {}
    for c in chunks:
        first_chunk_of_doc.setdefault(c["doc_id"], c["text"])  # chunks are ordered; chunk_0 first
    for doc_id, first_text in first_chunk_of_doc.items():
        if doc_id not in paper_metadata:
            print(f"[PaperMeta] Extracting title + model names for {doc_id}...")
            try:
                meta = llm.extract_paper_metadata(first_text)
                paper_metadata[doc_id] = {"title": meta.title, "models": meta.models}
                save_paper_metadata(paper_metadata)
            except Exception as e:
                print(f"[PaperMeta] Failed for {doc_id}: {e}. Re-run to resume.")
                return
    print("[PaperMeta] " + "; ".join(f"{d}: {m.get('models')}" for d, m in paper_metadata.items()))

    # 2b. Auto-extend the acronym dictionary from each paper's own "Full Name (ACR)" definitions.
    register_acronyms(extract_acronyms_from_texts([c["text"] for c in chunks]))

    # ── Extraction archive: reuse valid results, extract the rest ──
    fingerprint = extraction_fingerprint()
    if reextract:
        print("[Extract] --reextract: arsip lama diabaikan; semua chunk diekstrak ulang.")
        cache = {}
    else:
        try:
            cache = _load_extraction_cache(fingerprint)
        except ArtifactIntegrityError as e:
            print(f"\n[ERROR] {e}")
            return
    current_ids = {c["chunk_id"] for c in chunks}
    cache = {cid: e for cid, e in cache.items() if cid in current_ids}

    # 3. Extraction (archive saved after every chunk)
    pending_chunks = [c for c in chunks if not _reusable(cache.get(c["chunk_id"]), c["text"])]
    print(f"[Extract] {len(chunks) - len(pending_chunks)}/{len(chunks)} chunk dipakai ulang dari arsip; "
          f"{len(pending_chunks)} diekstrak.")

    for i, chunk in enumerate(pending_chunks, 1):
        print(f"Extracting knowledge from chunk {i}/{len(pending_chunks)} ({chunk['chunk_id']})...")
        try:
            paper_title = paper_metadata.get(chunk["doc_id"], {}).get("title")  # B1: title injection
            result = llm.extract_entities(chunk["text"], paper_title=paper_title)
        except Exception as e:
            print(f"\n[ERROR] Failed on chunk {chunk['chunk_id']}: {e}")
            print("[Checkpoint] Progress saved. Re-run 'python main.py build' to resume.\n")
            return
        cache[chunk["chunk_id"]] = {
            "text_sha256": _text_sha256(chunk["text"]), "status": result.status, "model": llm.last_model,
            "attempts": result.attempts, "dropped": result.dropped, "data": result.knowledge.model_dump(),
        }
        _save_extraction_cache(fingerprint, cache)

    _print_extraction_summary(cache)
    failed = [c["chunk_id"] for c in chunks if cache[c["chunk_id"]]["status"] == "failed"]
    if failed:
        # A failed chunk is NOT done: stop before dedup so the graph never silently misses it.
        print(f"\n[Extract] {len(failed)} chunk GAGAL (output tak terbaca dua kali): {failed}")
        print("[Extract] Build dihentikan sebelum dedup. Jalankan lagi 'python main.py build' untuk "
              "mengulang chunk itu saja.")
        return
    print("All chunks extracted successfully.")
    # Chunk order (not extraction order) so dedup decisions are deterministic across resumes.
    knowledge_items = [(c["chunk_id"], ExtractedKnowledge.model_validate(cache[c["chunk_id"]]["data"]))
                       for c in chunks]

    # 3.5 Post-extraction validation + author normalization (graph-only; does NOT touch
    #     the vector store). These mutate the ExtractedKnowledge objects in place, so the
    #     (chunk_id, knowledge) pairs in knowledge_items stay in sync. Deterministic → resume-safe.
    knowledge_list = [k for _, k in knowledge_items]
    knowledge_list = clean_knowledge_list(knowledge_list)
    knowledge_list = preprocess_authors(knowledge_list)

    # 4. Dedup (with checkpoint support)
    from src.extraction.deduplicator import (
        _load_dedup_checkpoint, _load_dedup_decision_models, dedup_resume_index, DEDUP_CHECKPOINT_PATH,
    )
    dedup = HybridDeduplicator(llm, embed)
    
    # Try to resume dedup from checkpoint
    saved_master_list, saved_name_mapping = _load_dedup_checkpoint()
    dedup_resume_idx = 0
    
    if saved_master_list is not None:
        # Reconstruct master_list with embeddings re-embedded (they're cheap, local)
        print("[Checkpoint] Resuming dedup from saved state...")
        for ent_type, entities in saved_master_list.items():
            dedup.master_list[ent_type] = {}
            for name, data in entities.items():
                dedup.master_list[ent_type][name] = {"embedding": data["embedding"]}
        dedup.name_mapping_global = saved_name_mapping
        dedup.decision_models = _load_dedup_decision_models()
        dedup.rebuild_normalized_index()
        dedup_resume_idx = dedup_resume_index(knowledge_list, saved_name_mapping)
        print(f"[Checkpoint] Dedup resuming from knowledge item {dedup_resume_idx}/{len(knowledge_list)}")

    print("Deduplicating entities...")
    try:
        dedup.process_extracted_knowledge(knowledge_list, resume_from_idx=dedup_resume_idx)
    except DailyQuotaExceededError as e:
        # Dedup deliberately has no fallback (merge decisions must come from one model). The checkpoint is already
        # saved by the deduplicator; a re-run after the quota resets continues from there.
        print(f"\n[Dedup] Stopped: {e}")
        print(f"[Checkpoint] Dedup progress saved to {DEDUP_CHECKPOINT_PATH}.")
        return
    # The dedup checkpoint is NOT deleted here, but at the end of the build together with the extraction checkpoint:
    # if the build fails after this point, a re-run skips dedup (every item is already mapped) instead of
    # repeating all the merge decisions (wasting quota, and the results could differ).

    # 5. Graph Construction (with provenance: anchors every entity to its source Paper)
    graph_builder = GraphBuilder()
    graph_builder.add_knowledge_with_provenance(knowledge_items)
    graph_builder.link_models_to_papers(paper_metadata)  # B2: link popular model names to papers
    graph_builder.reclassify_noncanonical_papers(paper_metadata)  # drop cited-paper leakage (e.g. NiN)
    graph_builder.link_entity_families()      # link versioned variants to a family node
    try:
        graph_builder.merge_family_aliases(llm)  # merge families that are mentioned differently (ChatGPT vs GPT)
    except DailyQuotaExceededError as e:
        _stop_for_quota("family merge", e)
        return
    
    # 6. Community Detection & Summarization
    analytics = GraphAnalytics(graph_builder.graph)
    print("Detecting communities...")
    communities = analytics.detect_communities()
    covered = sum(len(c['nodes']) for c in communities)
    print(f"{len(communities)} communities (>= {COMMUNITY_MIN_SIZE} nodes) covering {covered} nodes; "
          "the rest are outliers.")

    summaries = []
    for c in communities:
        print(f"Summarizing community {c['community_id']} ({len(c['nodes'])} nodes)...")
        comm_text = analytics.community_context(c['nodes'])
        try:
            summary = llm.summarize_community(comm_text)
            summaries.append({
                "community_id": c['community_id'],
                "summary": summary
            })
        except DailyQuotaExceededError as e:
            # Both Gemini AND Groq exhausted: stop, do not finish the build with incomplete summaries.
            _stop_for_quota("community summarization", e)
            return
        except Exception as e:
            print(f"[Warning] Skipped community {c['community_id']} summarization: {e}")

    # Saved AFTER the stages that can stop on quota: stopping midway must not
    # leave a new graph.graphml next to an old chroma_db/manifest (the engine refuses to start).
    graph_builder.save_graph("data/processed/graph.graphml")

    # 7. Vector DB Storage
    chroma = ChromaStore()
    
    print("Embedding and storing chunks...")
    chunk_texts = [c["text"] for c in chunks]
    chunk_embs = embed.embed_batch(chunk_texts)
    chroma.add_chunks(chunks, chunk_embs)
    
    # Always called (even when empty): the collection is replaced entirely, no summaries from an old build remain.
    print("Embedding and storing community summaries...")
    summary_embs = embed.embed_batch([s["summary"] for s in summaries]) if summaries else []
    chroma.add_community_summaries(summaries, summary_embs)

    # 7.5 Integrity manifest (embedding model + revision, graph SHA-256/counts, Chroma counts).
    #     The engine verifies it at startup. A failure here must not discard a completed build.
    try:
        write_manifest(build_manifest("data/processed/graph.graphml", graph_builder.graph, chroma))
        print("[Integrity] Wrote data/processed/manifest.json")
    except Exception as e:
        print(f"[Integrity] Could not write the manifest: {e}. Create it later with: python corpus.py manifest")

    # 7.6 Fact embedding cache for graph rerank (query-time): filled completely now so no query
    #     has to wait for embedding. Read from the saved graph.graphml so the texts are exactly what they are at query time.
    #     No LLM; a failure here does not cancel the build (a query fills the cache itself if needed).
    try:
        import networkx as nx
        stats = GraphFactReranker(embed, cache_path=FACT_EMBEDDINGS_PATH).warm(
            nx.read_graphml("data/processed/graph.graphml"))
        print(f"[Rerank] Fact embedding cache: {stats['facts']} facts, {stats['computed']} newly embedded, "
              f"{stats['pruned']} stale removed ({FACT_EMBEDDINGS_PATH})")
    except Exception as e:
        print(f"[Rerank] Could not fill the fact embedding cache: {e}. Queries will fill it on demand.")

    # Clean up checkpoints after successful completion
    # The extraction archive is kept on purpose (audit + reuse); only the dedup checkpoint goes.
    removed = [p for p in (DEDUP_CHECKPOINT_PATH,) if os.path.exists(p)]
    for p in removed:
        os.remove(p)
    if removed:
        print(f"[Checkpoint] Cleared the dedup checkpoint (build complete; extraction archive kept: "
              f"{EXTRACTION_CACHE_PATH}).")

    print("=== Build Pipeline Complete ===")

    # 8. Generate the standalone visualization snapshot (not auto-opened).
    _generate_visualization(graph_builder.graph)


# ──────────────────────────────────────────
# Query Pipeline
# ──────────────────────────────────────────
def query_pipeline(query: str):
    print("=== Starting Query ===")
    if not os.path.exists("data/processed/graph.graphml"):
        print("Graph not found. Please run 'build' first.")
        return
        
    try:
        engine = GraphRAGEngine("data/processed/graph.graphml")
    except ArtifactIntegrityError as e:
        print(f"\n[ERROR] {e}")
        return
    answer = engine.query(query)
    
    print("\n" + "="*40)
    print("Q:", query)
    print("-" * 40)
    print("A:", answer)
    print("="*40 + "\n")

USAGE = """
GraphRAG Paper Explorer
=======================

  python main.py build                      Bangun knowledge graph dari PDF di data/raw/
  python main.py build --reextract          Sama, tapi abaikan arsip ekstraksi (ekstrak ulang semua)
  python main.py query -q "pertanyaan"      Tanya lewat terminal

Perintah lain:
  python server.py                          Web app: visualisasi + chatbot (http://127.0.0.1:8000)
  python evaluate.py                        Ukur akurasi retrieval (gratis, tanpa LLM)
  python corpus.py list                     Kelola korpus (backup / clear / restore)

Contoh:
  python main.py build
  python main.py query -q "Metode apa yang paling banyak dipakai?"
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GraphRAG Paper Explorer CLI", add_help=True
    )
    # mode is optional so `python main.py` shows clear usage guidance,
    # rather than a confusing argparse error.
    parser.add_argument("mode", nargs="?", choices=["build", "query"],
                        help="build = bangun graph, query = tanya")
    parser.add_argument("--query", "-q", type=str, help="Pertanyaan (untuk mode 'query')")
    parser.add_argument("--reextract", action="store_true",
                        help="build: abaikan arsip ekstraksi dan ekstrak ulang semua chunk")

    args = parser.parse_args()

    if args.mode is None:
        print(USAGE)
    elif args.mode == "build":
        build_pipeline(reextract=args.reextract)
    elif args.mode == "query":
        if not args.query:
            print('Error: mode "query" butuh pertanyaan.\n')
            print('  Contoh: python main.py query -q "Metode apa yang paling banyak dipakai?"')
        else:
            query_pipeline(args.query)
