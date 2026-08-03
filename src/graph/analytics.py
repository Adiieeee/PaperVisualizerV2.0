import networkx as nx
from networkx.algorithms.community import louvain_communities
from typing import List, Dict, Any

class GraphAnalytics:
    def __init__(self, graph: nx.Graph):
        self.graph = graph

    def get_counting_stats(self, target_type: str) -> List[tuple]:
        """
        Returns a sorted list of (node_name, degree) for nodes of a specific type.
        This handles the "Counting/Ranking" intent natively without LLM.

        `mentioned_in` provenance edges (added to anchor orphan nodes to their source
        Paper) are excluded from the degree count so they don't inflate the ranking —
        we want "most connected by real relationships", not "appears in most papers".
        """
        nodes_of_type = [n for n, d in self.graph.nodes(data=True) if d.get('type') == target_type]

        degrees = []
        for n in nodes_of_type:
            deg = sum(
                1 for _, _, ed in self.graph.edges(n, data=True)
                if ed.get('relation') not in ('mentioned_in', 'variant_of')
            )
            degrees.append((n, deg))

        # Sort descending by degree
        sorted_degrees = sorted(degrees, key=lambda x: x[1], reverse=True)
        return sorted_degrees

    def get_bridge_entities(self, min_papers: int = 2, top_n: int = 15) -> List[tuple]:
        """
        Cross-document 'bridges': non-Paper entities directly connected to >= min_papers
        Paper nodes (i.e. entities that appear in multiple papers). Returns
        [(entity_name, [paper_names], paper_count)] sorted by paper_count desc.
        This answers structural questions no pure-vector RAG can ("what connects the papers?").
        """
        papers = {n for n, d in self.graph.nodes(data=True) if d.get("type") == "Paper"}
        results = []
        for n, d in self.graph.nodes(data=True):
            if d.get("type") == "Paper":
                continue
            connected = [nb for nb in self.graph.neighbors(n) if nb in papers]
            if len(connected) >= min_papers:
                results.append((n, sorted(connected), len(connected)))
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_n]

    def detect_communities(self) -> List[Dict[str, Any]]:
        """
        Detects communities using Louvain.
        CRITICAL: Excludes 'Paper' nodes to prevent them from acting as artificial hubs.
        Returns a list of dicts: [{'community_id': int, 'nodes': List[str]}]
        """
        # Create a view/subgraph without 'Paper' nodes
        non_paper_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('type') != 'Paper']
        analysis_graph = self.graph.subgraph(non_paper_nodes)
        
        # Run native NetworkX Louvain
        communities_sets = louvain_communities(analysis_graph)
        
        results = []
        for i, comm_set in enumerate(communities_sets):
            results.append({
                "community_id": i,
                "nodes": list(comm_set)
            })
            
        return results
