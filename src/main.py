import argparse
import os

from src.ingestion.document import DocumentProcessor
from src.llm.gemini_client import GeminiClient
from src.llm.embedding import EmbeddingClient
from src.extraction.extractor import KnowledgeExtractor
from src.extraction.deduplicator import HybridDeduplicator
from src.graph.builder import GraphBuilder
from src.graph.analytics import GraphAnalytics
from src.vectorstore.chromadb_store import ChromaStore
from src.engine import GraphRAGEngine

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
    
    # 3. Extraction
    extractor = KnowledgeExtractor(llm)
    # FOR TESTING: we can limit chunks to save API calls initially
    # chunks = chunks[:5]
    knowledge_list = extractor.extract_from_chunks(chunks)
    
    # 4. Dedup
    dedup = HybridDeduplicator(llm, embed)
    print("Deduplicating entities...")
    dedup.process_extracted_knowledge(knowledge_list)
    
    # 5. Graph Construction
    graph_builder = GraphBuilder()
    graph_builder.add_knowledge(knowledge_list)
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
            summary = llm.summarize_community(comm_text)
            summaries.append({
                "community_id": c['community_id'],
                "summary": summary
            })
            
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
        
    print("=== Build Pipeline Complete ===")

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GraphRAG Paper Explorer CLI")
    parser.add_argument("mode", choices=["build", "query"], help="Mode of operation")
    parser.add_argument("--query", "-q", type=str, help="The query string (required for 'query' mode)")
    
    args = parser.parse_args()
    
    if args.mode == "build":
        build_pipeline()
    elif args.mode == "query":
        if not args.query:
            print("Error: --query is required when mode is 'query'")
        else:
            query_pipeline(args.query)
