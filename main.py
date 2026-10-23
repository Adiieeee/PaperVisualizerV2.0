import argparse
import os
import json

from src.ingestion.document import DocumentProcessor
from src.llm.llm_service import LLMService, DailyQuotaExceededError
from src.llm.embedding import EmbeddingClient
from src.extraction.cleaner import clean_knowledge_list, preprocess_authors
from src.extraction.deduplicator import HybridDeduplicator, extract_acronyms_from_texts, register_acronyms
from src.graph.builder import GraphBuilder
from src.graph.analytics import GraphAnalytics
from src.vectorstore.chromadb_store import ChromaStore
from src.engine import GraphRAGEngine
from src.config import ExtractedKnowledge, load_paper_metadata, save_paper_metadata
from src.security.integrity import (
    ArtifactIntegrityError, atomic_write_json, build_manifest, chroma_stats, write_manifest,
)

# ──────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────
CHECKPOINT_PATH = "data/processed/checkpoint_extraction.json"
# Model yang mengekstrak tiap chunk (Gemini, atau Groq saat kuota harian Gemini habis). Checkpoint
# dihapus setelah build sukses, jadi mapping ini juga ditulis permanen untuk audit/proses ulang.
EXTRACTION_MODELS_PATH = "data/processed/extraction_models.json"

def _save_checkpoint(done_chunk_ids: set, knowledge_items: list, chunk_models: dict = None):
    """
    Save progress to disk. knowledge_items is a list of (chunk_id, ExtractedKnowledge).
    The chunk_id is persisted per item so provenance survives a resume (Phase A) —
    Pydantic private attrs are dropped by model_dump(), so we track it at the item level.
    chunk_models: {chunk_id: model yang menjawab}; dibaca lagi lewat _load_chunk_models().
    """
    payload = {
        "done_chunk_ids": list(done_chunk_ids),
        "knowledge": [{"chunk_id": cid, "data": k.model_dump()} for cid, k in knowledge_items],
        "models": chunk_models or {},
    }
    # Atomic (temp file + os.replace): this file holds costly LLM output; a crash mid-write must
    # never leave truncated JSON that makes the build impossible to resume.
    atomic_write_json(CHECKPOINT_PATH, payload, indent=2)

def _load_checkpoint():
    """Load checkpoint if present. Returns (done_chunk_ids, knowledge_items) where each
    item is (chunk_id, ExtractedKnowledge). Tolerant of the old format (no chunk_id)."""
    if not os.path.exists(CHECKPOINT_PATH):
        return set(), []
    try:
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except ValueError as e:  # corrupt JSON: do NOT silently restart and burn the quota again
        raise ArtifactIntegrityError(
            f"{CHECKPOINT_PATH} is corrupt ({e}). Fix it by hand, or move it aside "
            f"(e.g. rename to .corrupt) to re-extract from scratch."
        ) from e
    items = []
    for entry in payload["knowledge"]:
        if isinstance(entry, dict) and "data" in entry:  # v2 format
            items.append((entry.get("chunk_id"), ExtractedKnowledge.model_validate(entry["data"])))
        else:  # legacy format: a bare knowledge dict, no chunk_id
            items.append((None, ExtractedKnowledge.model_validate(entry)))
    return set(payload["done_chunk_ids"]), items

def _vector_store_mismatch(chroma, embed):
    """Pesan error bila ChromaDB berisi vektor berdimensi lain dari model embedding sekarang, else None."""
    stored = chroma_stats(chroma)["dimension"]
    if stored is None:  # store kosong: aman
        return None
    current = len(embed.embed_text("dimension probe"))
    if stored == current:
        return None
    return (f"ChromaDB berisi vektor {stored} dimensi, model embedding sekarang menghasilkan {current} "
            f"dimensi (model diganti?). Build dihentikan sebelum memakai kuota LLM. Kosongkan indeks "
            f"dulu: `python corpus.py reset-index` (PDF dan paper_metadata.json dipertahankan).")


def _stop_for_quota(stage: str, exc: Exception):
    """Kuota harian habis setelah dedup. Checkpoint ekstraksi + dedup masih ada, jadi re-run
    setelah reset melewati keduanya dan mengulang hanya tahap graph/komunitas."""
    print(f"\n[Build] Stopped during {stage}: {exc}")
    print("[Checkpoint] Extraction + dedup checkpoints kept; re-run 'python main.py build' after the "
          "quota resets (graph, family merge and community summaries are redone).")


def _load_chunk_models() -> dict:
    """{chunk_id: model} dari checkpoint (kosong untuk checkpoint lama yang belum mencatatnya).
    Dipanggil setelah _load_checkpoint, yang sudah menolak JSON korup."""
    if not os.path.exists(CHECKPOINT_PATH):
        return {}
    with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
        return json.load(f).get("models", {})


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
def build_pipeline():
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

    # ── Checkpoint: load progress ──
    # knowledge_items: list of (chunk_id, ExtractedKnowledge) — chunk_id carries provenance.
    try:
        done_chunk_ids, knowledge_items = _load_checkpoint()
    except ArtifactIntegrityError as e:
        print(f"\n[ERROR] {e}")
        return
    chunk_models = _load_chunk_models()
    if done_chunk_ids:
        print(f"[Checkpoint] Resuming. {len(done_chunk_ids)}/{len(chunks)} chunks already done.")

    # 3. Extraction (with per-chunk checkpoint saves)
    pending_chunks = [c for c in chunks if c["chunk_id"] not in done_chunk_ids]
    print(f"Remaining chunks to extract: {len(pending_chunks)}")

    initial_done_count = len(done_chunk_ids)
    for i, chunk in enumerate(pending_chunks):
        global_idx = initial_done_count + i + 1
        print(f"Extracting knowledge from chunk {global_idx}/{len(chunks)} ({chunk['chunk_id']})...")
        try:
            paper_title = paper_metadata.get(chunk["doc_id"], {}).get("title")  # B1: title injection
            knowledge = llm.extract_entities(chunk["text"], paper_title=paper_title).knowledge
            knowledge_items.append((chunk["chunk_id"], knowledge))
            done_chunk_ids.add(chunk["chunk_id"])
            chunk_models[chunk["chunk_id"]] = llm.last_model
            # Save checkpoint after EVERY successful extraction
            _save_checkpoint(done_chunk_ids, knowledge_items, chunk_models)
        except Exception as e:
            print(f"\n[ERROR] Failed on chunk {chunk['chunk_id']}: {e}")
            print("[Checkpoint] Progress saved. Re-run 'python main.py build' to resume.\n")
            return

    print("All chunks extracted successfully.")
    atomic_write_json(EXTRACTION_MODELS_PATH, chunk_models, indent=2)
    model_counts = {}
    for m in chunk_models.values():
        model_counts[m] = model_counts.get(m, 0) + 1
    print(f"[LLM] Chunks per model: {model_counts}  (detail: {EXTRACTION_MODELS_PATH})")

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
        # Dedup sengaja tanpa fallback (keputusan merge harus dari satu model). Checkpoint sudah
        # disimpan oleh deduplicator; re-run setelah kuota reset melanjutkan dari sana.
        print(f"\n[Dedup] Stopped: {e}")
        print(f"[Checkpoint] Dedup progress saved to {DEDUP_CHECKPOINT_PATH}.")
        return
    # Checkpoint dedup TIDAK dihapus di sini, tapi di akhir build bersama checkpoint ekstraksi:
    # kalau build gagal sesudah ini, re-run melewati dedup (semua item sudah terpetakan) alih-alih
    # mengulang semua keputusan merge (boros kuota, dan hasilnya bisa berbeda).

    # 5. Graph Construction (with provenance: anchors every entity to its source Paper)
    graph_builder = GraphBuilder()
    graph_builder.add_knowledge_with_provenance(knowledge_items)
    graph_builder.link_models_to_papers(paper_metadata)  # B2: link popular model names to papers
    graph_builder.reclassify_noncanonical_papers(paper_metadata)  # drop cited-paper leakage (e.g. NiN)
    graph_builder.link_entity_families()      # hubungkan varian ber-versi ke node keluarga
    try:
        graph_builder.merge_family_aliases(llm)  # satukan keluarga yg beda penyebutan (ChatGPT vs GPT)
    except DailyQuotaExceededError as e:
        _stop_for_quota("family merge", e)
        return
    
    # 6. Community Detection & Summarization
    analytics = GraphAnalytics(graph_builder.graph)
    print("Detecting communities...")
    communities = analytics.detect_communities()
    
    summaries = []
    for c in communities:
        if len(c['nodes']) > 1:
            print(f"Summarizing community {c['community_id']} ({len(c['nodes'])} nodes)...")
            comm_text = ", ".join(c['nodes'])
            try:
                summary = llm.summarize_community(comm_text)
                summaries.append({
                    "community_id": c['community_id'],
                    "summary": summary
                })
            except DailyQuotaExceededError as e:
                # Gemini DAN Groq habis: berhenti, jangan selesaikan build dengan ringkasan bolong.
                _stop_for_quota("community summarization", e)
                return
            except Exception as e:
                print(f"[Warning] Skipped community {c['community_id']} summarization: {e}")

    # Disimpan SETELAH tahap yang bisa berhenti karena kuota: berhenti di tengah tidak boleh
    # meninggalkan graph.graphml baru di samping chroma_db/manifest lama (engine menolak start).
    graph_builder.save_graph("data/processed/graph.graphml")

    # 7. Vector DB Storage
    chroma = ChromaStore()
    
    print("Embedding and storing chunks...")
    chunk_texts = [c["text"] for c in chunks]
    chunk_embs = embed.embed_batch(chunk_texts)
    chroma.add_chunks(chunks, chunk_embs)
    
    if summaries:
        print("Embedding and storing community summaries...")
        summary_texts = [s["summary"] for s in summaries]
        summary_embs = embed.embed_batch(summary_texts)
        chroma.add_community_summaries(summaries, summary_embs)

    # 7.5 Integrity manifest (embedding model + revision, graph SHA-256/counts, Chroma counts).
    #     The engine verifies it at startup. A failure here must not discard a completed build.
    try:
        write_manifest(build_manifest("data/processed/graph.graphml", graph_builder.graph, chroma))
        print("[Integrity] Wrote data/processed/manifest.json")
    except Exception as e:
        print(f"[Integrity] Could not write the manifest: {e}. Create it later with: python corpus.py manifest")

    # Clean up checkpoints after successful completion
    removed = [p for p in (CHECKPOINT_PATH, DEDUP_CHECKPOINT_PATH) if os.path.exists(p)]
    for p in removed:
        os.remove(p)
    if removed:
        print("[Checkpoint] Cleared all checkpoints (build complete).")

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
    # mode dibuat opsional supaya `python main.py` menampilkan panduan yang jelas,
    # bukan error argparse yang membingungkan.
    parser.add_argument("mode", nargs="?", choices=["build", "query"],
                        help="build = bangun graph, query = tanya")
    parser.add_argument("--query", "-q", type=str, help="Pertanyaan (untuk mode 'query')")

    args = parser.parse_args()

    if args.mode is None:
        print(USAGE)
    elif args.mode == "build":
        build_pipeline()
    elif args.mode == "query":
        if not args.query:
            print('Error: mode "query" butuh pertanyaan.\n')
            print('  Contoh: python main.py query -q "Metode apa yang paling banyak dipakai?"')
        else:
            query_pipeline(args.query)
