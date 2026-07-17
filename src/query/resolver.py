import networkx as nx
from typing import List, Dict, Any, Tuple
import difflib

from src.llm.embedding import EmbeddingClient
from src.vectorstore.chromadb_store import ChromaStore

class QueryResolver:
    def __init__(self, embed_client: EmbeddingClient, chroma_store: ChromaStore, graph: nx.Graph):
        self.embed = embed_client
        self.chroma = chroma_store
        self.graph = graph

    def resolve_entity(self, query: str) -> str:
        """
        Hybrid resolution for entity search.
        1. Exact match on graph nodes
        2. Fuzzy match on graph nodes (SequenceMatcher)
        3. If we had an alias dictionary, we'd use it here.
        Returns the canonical node name, or None if not found.
        """
        nodes = list(self.graph.nodes())
        
        # 1. Exact match (case insensitive)
        query_lower = query.lower()
        for node in nodes:
            if node.lower() == query_lower:
                return node
                
        # 2. Fuzzy matching (SequenceMatcher)
        best_match = None
        highest_ratio = 0.0
        for node in nodes:
            ratio = difflib.SequenceMatcher(None, query_lower, node.lower()).ratio()
            if ratio > highest_ratio:
                highest_ratio = ratio
                best_match = node
                
        # Threshold for fuzzy match
        if highest_ratio > 0.8:
            print(f"Fuzzy matched '{query}' to '{best_match}' (Ratio: {highest_ratio:.2f})")
            return best_match
            
        print(f"Could not resolve entity '{query}' directly in the graph.")
        return None

    def search_chunks(self, query: str, top_k: int = 5) -> List[str]:
        """
        Vector search on raw chunks as a fallback.
        """
        emb = self.embed.embed_text(query)
        results = self.chroma.search_chunks(emb, n_results=top_k)
        
        if not results['documents'] or not results['documents'][0]:
            return []
            
        return results['documents'][0]

    def search_communities(self, query: str, top_k: int = 3) -> List[str]:
        """
        Vector search on community summaries for thematic queries.
        """
        emb = self.embed.embed_text(query)
        results = self.chroma.search_communities(emb, n_results=top_k)
        
        if not results['documents'] or not results['documents'][0]:
            return []
            
        return results['documents'][0]
