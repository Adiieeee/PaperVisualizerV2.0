import argparse
import os
import json

from src.ingestion.document import DocumentProcessor
from src.llm.gemini_client import GeminiClient
from src.llm.embedding import EmbeddingClient
from src.extraction.cleaner import clean_knowledge_list, preprocess_authors
from src.extraction.deduplicator import HybridDeduplicator, extract_acronyms_from_texts, register_acronyms
from src.graph.builder import GraphBuilder
from src.graph.analytics import GraphAnalytics
from src.vectorstore.chromadb_store import ChromaStore
from src.engine import GraphRAGEngine
from src.config import ExtractedKnowledge, load_paper_metadata, save_paper_metadata

# ──────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────
CHECKPOINT_PATH = "data/processed/checkpoint_extraction.json"

def _save_checkpoint(done_chunk_ids: set, knowledge_items: list):
    """
    Save progress to disk. knowledge_items is a list of (chunk_id, ExtractedKnowledge).
    The chunk_id is persisted per item so provenance survives a resume (Phase A) —
    Pydantic private attrs are dropped by model_dump(), so we track it at the item level.
    """
    os.makedirs("data/processed", exist_ok=True)
    payload = {
        "done_chunk_ids": list(done_chunk_ids),
        "knowledge": [{"chunk_id": cid, "data": k.model_dump()} for cid, k in knowledge_items],
    }
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def _load_checkpoint():
    """Load checkpoint if present. Returns (done_chunk_ids, knowledge_items) where each
    item is (chunk_id, ExtractedKnowledge). Tolerant of the old format (no chunk_id)."""
    if not os.path.exists(CHECKPOINT_PATH):
        return set(), []
    with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)
    items = []
    for entry in payload["knowledge"]:
        if isinstance(entry, dict) and "data" in entry:  # v2 format
            items.append((entry.get("chunk_id"), ExtractedKnowledge.model_validate(entry["data"])))
        else:  # legacy format: a bare knowledge dict, no chunk_id
            items.append((None, ExtractedKnowledge.model_validate(entry)))
    return set(payload["done_chunk_ids"]), items


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
    llm = GeminiClient()
    embed = EmbeddingClient()

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
    done_chunk_ids, knowledge_items = _load_checkpoint()
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
            knowledge = llm.extract_entities(chunk["text"], paper_title=paper_title)
            knowledge_items.append((chunk["chunk_id"], knowledge))
            done_chunk_ids.add(chunk["chunk_id"])
            # Save checkpoint after EVERY successful extraction
            _save_checkpoint(done_chunk_ids, knowledge_items)
        except Exception as e:
            print(f"\n[ERROR] Failed on chunk {chunk['chunk_id']}: {e}")
            print("[Checkpoint] Progress saved. Re-run 'python main.py build' to resume.\n")
            return

    print("All chunks extracted successfully.")

    # 3.5 Post-extraction validation + author normalization (graph-only; does NOT touch
    #     the vector store). These mutate the ExtractedKnowledge objects in place, so the
    #     (chunk_id, knowledge) pairs in knowledge_items stay in sync. Deterministic → resume-safe.
    knowledge_list = [k for _, k in knowledge_items]
    knowledge_list = clean_knowledge_list(knowledge_list)
    knowledge_list = preprocess_authors(knowledge_list)

    # 4. Dedup (with checkpoint support)
    from src.extraction.deduplicator import _load_dedup_checkpoint, DEDUP_CHECKPOINT_PATH
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
        # Count how many knowledge items were fully processed
        dedup_resume_idx = sum(
            1 for k in knowledge_list
            if all(ent.name in saved_name_mapping for ent in k.entities)
        )
        print(f"[Checkpoint] Dedup resuming from knowledge item ~{dedup_resume_idx}/{len(knowledge_list)}")

    print("Deduplicating entities...")
    dedup.process_extracted_knowledge(knowledge_list, resume_from_idx=dedup_resume_idx)
    
    # Clean up dedup checkpoint after success
    if os.path.exists(DEDUP_CHECKPOINT_PATH):
        os.remove(DEDUP_CHECKPOINT_PATH)
    
    # 5. Graph Construction (with provenance: anchors every entity to its source Paper)
    graph_builder = GraphBuilder()
    graph_builder.add_knowledge_with_provenance(knowledge_items)
    graph_builder.link_models_to_papers(paper_metadata)  # B2: link popular model names to papers
    graph_builder.reclassify_noncanonical_papers(paper_metadata)  # drop cited-paper leakage (e.g. NiN)
    graph_builder.link_entity_families()      # hubungkan varian ber-versi ke node keluarga
    graph_builder.merge_family_aliases(llm)   # satukan keluarga yg beda penyebutan (ChatGPT vs GPT)
    graph_builder.save_graph("data/processed/graph.graphml")
    
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
            except Exception as e:
                print(f"[Warning] Skipped community {c['community_id']} summarization: {e}")
            
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
        
    # Clean up extraction checkpoint after successful completion
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
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
        
    engine = GraphRAGEngine("data/processed/graph.graphml")
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
