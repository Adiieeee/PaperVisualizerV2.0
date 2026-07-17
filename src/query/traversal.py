import networkx as nx
from typing import List, Dict, Any

class GraphTraversal:
    def __init__(self, graph: nx.Graph):
        self.graph = graph

    def traverse_1_hop(self, start_node: str) -> Dict[str, Any]:
        """
        Performs a 1-hop traversal from the start_node.
        Implements Paper-blocking: neighbors of type 'Paper' are included as provenance
        but we explicitly do not traverse *through* them in further hops (if multi-hop was implemented).
        Here, it returns the 1-hop context.
        """
        if not self.graph.has_node(start_node):
            return {"error": f"Node '{start_node}' not found in the graph."}

        node_data = self.graph.nodes[start_node]
        context = {
            "entity": start_node,
            "type": node_data.get("type"),
            "description": node_data.get("description"),
            "relationships": []
        }

        # 1-hop neighbors
        for neighbor in self.graph.neighbors(start_node):
            edge_data = self.graph.get_edge_data(start_node, neighbor)
            neighbor_data = self.graph.nodes[neighbor]
            
            rel_info = {
                "target": neighbor,
                "target_type": neighbor_data.get("type"),
                "relation": edge_data.get("relation"),
                "description": edge_data.get("description")
            }
            context["relationships"].append(rel_info)
            
            # Paper blocking logic note:
            # Since this is strictly 1-hop, we just collect it. 
            # If we were doing 2-hops, we'd add: if neighbor_data.get("type") == "Paper": continue

        return context

    def format_traversal_context(self, traversal_result: Dict[str, Any]) -> str:
        """
        Formats the traversal dictionary into a readable string for the LLM.
        """
        if "error" in traversal_result:
            return traversal_result["error"]
            
        lines = []
        lines.append(f"Entity: {traversal_result['entity']} ({traversal_result['type']})")
        if traversal_result['description']:
            lines.append(f"Description: {traversal_result['description']}")
            
        lines.append("Relationships:")
        for rel in traversal_result['relationships']:
            lines.append(f"- {rel['relation']} -> {rel['target']} ({rel['target_type']})")
            if rel['description']:
                lines.append(f"  Note: {rel['description']}")
                
        return "\n".join(lines)
