import networkx as nx
from typing import Dict, Any


class GraphTraversal:
    def __init__(self, graph: nx.Graph):
        self.graph = graph

    def traverse(self, start_node: str, max_hops: int = 2, max_rels: int = 150) -> Dict[str, Any]:
        """
        BFS up to `max_hops` from start_node (default 2 → enables lineage/chain answers and
        Paper→Author→Institution affiliation lookups).

        Paper-blocking: Paper nodes are still recorded as neighbors (provenance) but are NOT
        expanded further — this keeps the context from exploding through paper hubs while still
        letting method→method chains (e.g. GoogLeNet → Inception → Network in Network) surface.
        BOTH hops always run (traversal is cheap on this graph); only the RECORDED relationships
        are capped at `max_rels` so a high-degree hub can't flood the LLM context.
        """
        if not self.graph.has_node(start_node):
            return {"error": f"Node '{start_node}' not found in the graph."}

        node_data = self.graph.nodes[start_node]
        context = {
            "entity": start_node,
            "type": node_data.get("type"),
            "description": node_data.get("description"),
            "relationships": [],
        }

        visited_nodes = {start_node}
        visited_edges = set()
        frontier = [start_node]

        for hop in range(1, max_hops + 1):
            next_frontier = []
            for cur in frontier:
                for nb in self.graph.neighbors(cur):
                    ekey = frozenset((cur, nb))
                    if ekey not in visited_edges:
                        visited_edges.add(ekey)
                        if len(context["relationships"]) < max_rels:
                            ed = self.graph.get_edge_data(cur, nb)
                            context["relationships"].append({
                                "source": cur,
                                "target": nb,
                                "target_type": self.graph.nodes[nb].get("type"),
                                "relation": ed.get("relation"),
                                "description": ed.get("description"),
                                "hop": hop,
                            })
                    if nb not in visited_nodes:
                        visited_nodes.add(nb)
                        # Paper-blocking: don't traverse THROUGH paper nodes.
                        if self.graph.nodes[nb].get("type") != "Paper":
                            next_frontier.append(nb)
            frontier = next_frontier

        return context

    # Backwards-compatible 1-hop helper.
    def traverse_1_hop(self, start_node: str) -> Dict[str, Any]:
        return self.traverse(start_node, max_hops=1)

    def format_traversal_context(self, traversal_result: Dict[str, Any]) -> str:
        """Format the (possibly multi-hop) traversal into readable text for the LLM."""
        if "error" in traversal_result:
            return traversal_result["error"]

        lines = [f"Entity: {traversal_result['entity']} ({traversal_result['type']})"]
        if traversal_result.get("description"):
            lines.append(f"Description: {traversal_result['description']}")

        lines.append("Relationships (direct = hop 1, indirect chain = hop 2+):")
        for rel in traversal_result["relationships"]:
            lines.append(
                f"- [hop {rel['hop']}] {rel['source']} --{rel['relation']}--> "
                f"{rel['target']} ({rel.get('target_type')})"
            )
            if rel.get("description"):
                lines.append(f"    Note: {rel['description']}")

        return "\n".join(lines)
