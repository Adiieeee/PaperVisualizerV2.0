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
        """
        nodes_of_type = [n for n, d in self.graph.nodes(data=True) if d.get('type') == target_type]
        subgraph = self.graph.subgraph(nodes_of_type)
        
        # Calculate degree (number of connections) in the full graph for these nodes
        degrees = [(n, self.graph.degree(n)) for n in nodes_of_type]
        
        # Sort descending by degree
        sorted_degrees = sorted(degrees, key=lambda x: x[1], reverse=True)
        return sorted_degrees

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
