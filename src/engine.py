import networkx as nx

from src.llm.gemini_client import GeminiClient
from src.llm.embedding import EmbeddingClient
from src.vectorstore.chromadb_store import ChromaStore
from src.query.router import QueryRouter
from src.query.resolver import QueryResolver
from src.query.traversal import GraphTraversal
from src.graph.analytics import GraphAnalytics

class GraphRAGEngine:
    def __init__(self, graph_path: str = "data/processed/graph.graphml"):
        self.llm = GeminiClient()
        self.embed = EmbeddingClient()
        self.chroma = ChromaStore()
        
        # Load Graph
        self.graph = nx.read_graphml(graph_path)
        print(f"Engine loaded graph with {self.graph.number_of_nodes()} nodes.")
        
        self.router = QueryRouter(self.llm)
        self.resolver = QueryResolver(self.embed, self.chroma, self.graph)
        self.traversal = GraphTraversal(self.graph)
        self.analytics = GraphAnalytics(self.graph)

    def query(self, user_query: str) -> str:
        """
        End-to-end query processing pipeline.
        """
        print(f"\n[1] Routing query: '{user_query}'")
        routing = self.router.route(user_query)
        
        context_parts = []
        
        if routing.intent == "counting":
            print(f"[2] Executing Graph Stats for '{routing.target_type}'")
            if not routing.target_type:
                return "Error: Could not determine which entity type to count."
                
            stats = self.analytics.get_counting_stats(routing.target_type)
            # Take top 10 to avoid context overflow
            top_stats = stats[:10]
            context_parts.append(f"Top {routing.target_type}s by number of connections:")
            for node, degree in top_stats:
                context_parts.append(f"- {node}: {degree} connections")
                
        elif routing.intent == "thematic":
            print("[2] Executing Vector Search on Community Summaries")
            search_q = routing.search_query or user_query
            summaries = self.resolver.search_communities(search_q)
            context_parts.append("Thematic Community Context:")
            for i, s in enumerate(summaries):
                context_parts.append(f"[{i+1}] {s}")
                
        else: # entity
            print("[2] Executing Hybrid Entity Search")
            search_q = routing.search_query or user_query
            resolved_node = self.resolver.resolve_entity(search_q)
            
            if resolved_node:
                print(f"[3] Traversing graph from '{resolved_node}'")
                trav_result = self.traversal.traverse_1_hop(resolved_node)
                context_parts.append("Graph Context:")
                context_parts.append(self.traversal.format_traversal_context(trav_result))
            else:
                print("[3] Node not found, falling back to Chunk search")
                chunks = self.resolver.search_chunks(search_q)
                context_parts.append("Raw Text Chunks Context:")
                for i, c in enumerate(chunks):
                    context_parts.append(f"[{i+1}] {c}")
                    
        # Assemble Context
        assembled_context = "\n\n".join(context_parts)
        print("\n[4] Assembled Context Length:", len(assembled_context))
        
        # Generate Answer
        print("[5] Generating Final Answer...")
        final_answer = self.llm.generate_final_answer(user_query, assembled_context)
        
        return final_answer
