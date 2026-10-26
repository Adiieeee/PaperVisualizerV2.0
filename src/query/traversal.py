import networkx as nx
from typing import Any, Dict, Optional


class GraphTraversal:
    def __init__(self, graph: nx.Graph):
        self.graph = graph

    def traverse(self, start_node: str, max_hops: int = 2, max_rels: Optional[int] = 150,
                 through_source_paper: bool = True) -> Dict[str, Any]:
        """
        BFS up to `max_hops` from start_node (default 2 → enables lineage/chain answers and
        Paper→Author→Institution affiliation lookups).

        Paper-blocking: Paper nodes are still recorded as neighbors (provenance) but are NOT
        expanded further — this keeps the context from exploding through paper hubs while still
        letting method→method chains (e.g. GoogLeNet → Inception → Network in Network) surface.
        BOTH hops always run (traversal is cheap on this graph); only the RECORDED relationships
        are capped at `max_rels` so a high-degree hub can't flood the LLM context.

        `through_source_paper` (roadmap 1.1): exempt the SOURCE paper from Paper-blocking, i.e. the
        paper directly adjacent to the start node. Reason: many paper-level facts
        (`uses_dataset`, `has_limitation`, `finds`) attach to the Paper node, while the user asks
        using the entity's name (e.g. "dataset apa yang dipakai SycEval?" although SYCEval is a
        Dataset node that only points to its paper). Papers found at hop >1 stay
        blocked so the context does not explode through cross-paper hubs.
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

        # Budget PER HOP. Without this, hub nodes (a paper with dozens of direct relations)
        # use up all of `max_rels` at hop-1, so hop-2 facts like
        # Paper -> Author -> Institution are never recorded.
        # max_rels=None: record every relation; pruning is left to GraphFactReranker (by relevance).
        if max_rels is None:
            max_rels = hop1_cap = float("inf")
        else:
            hop1_cap = max(1, int(max_rels * 0.6))
        hop_used = {}

        for hop in range(1, max_hops + 1):
            cap = hop1_cap if hop == 1 else max_rels
            next_frontier = []
            for cur in frontier:
                for nb in self.graph.neighbors(cur):
                    ekey = frozenset((cur, nb))
                    if ekey not in visited_edges:
                        visited_edges.add(ekey)
                        hop_used[hop] = hop_used.get(hop, 0)
                        if hop_used[hop] < cap and len(context["relationships"]) < max_rels:
                            hop_used[hop] += 1
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
                        is_paper = self.graph.nodes[nb].get("type") == "Paper"
                        # Paper-blocking, with one exception: the SOURCE paper (hop 1 from
                        # a non-Paper start node) may be expanded so paper-level facts are reachable.
                        is_source_paper = (
                            through_source_paper and is_paper and hop == 1
                            and node_data.get("type") != "Paper"
                        )
                        if not is_paper or is_source_paper:
                            next_frontier.append(nb)
            frontier = next_frontier

        return context

    def format_traversal_context(self, traversal_result: Dict[str, Any]) -> str:
        """
        Turn the traversal result into text for the LLM (roadmap 3.6).

        The old format was a flat list that repeated the entity name on EVERY line
        (~40 chars × 150 lines wasted) and mixed all relation types, so the LLM struggled
        to find specific facts. This format:
          - groups hop-1 relations PER RELATION TYPE (uses_dataset, finds, ...)
          - drops the repeated entity names
          - presents hop-2 relations as explicit CHAINS (A -> relation -> B) so
            multi-hop reasoning like Paper->Author->Institution is easy to see
        """
        if "error" in traversal_result:
            return traversal_result["error"]

        entity = traversal_result["entity"]
        out = [f"ENTITAS: {entity} ({traversal_result['type']})"]
        if traversal_result.get("description"):
            out.append(f"Deskripsi: {traversal_result['description']}")

        direct, indirect = {}, []
        for rel in traversal_result["relationships"]:
            target, desc = rel["target"], rel.get("description")
            if rel["hop"] == 1 and rel["source"] == entity:
                item = target if not desc else f"{target} — {desc}"
                direct.setdefault(rel["relation"] or "terkait", []).append(item)
            else:
                indirect.append(f"{rel['source']} → {rel['relation']} → {target}")

        if direct:
            out.append("\n--- RELASI LANGSUNG ---")
            for relation, items in direct.items():
                # Short names are joined on one line; long items (Finding/Limitation) go one per line
                if all(len(i) <= 55 for i in items):
                    out.append(f"▸ {relation} ({len(items)}): " + " · ".join(items))
                else:
                    out.append(f"▸ {relation} ({len(items)}):")
                    out.extend(f"   • {i}" for i in items)

        if indirect:
            out.append("\n--- RELASI TIDAK LANGSUNG (rantai 2 hop) ---")
            out.extend(f"▸ {c}" for c in indirect)

        return "\n".join(out)
