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

        `through_source_paper` (roadmap 1.1): kecualikan paper SUMBER dari Paper-blocking, yaitu
        paper yang bertetangga langsung dengan node awal. Alasannya: banyak fakta tingkat-paper
        (`uses_dataset`, `has_limitation`, `finds`) menempel di node Paper, sementara user bertanya
        memakai nama entitasnya (mis. "dataset apa yang dipakai SycEval?" padahal SYCEval adalah
        node Dataset yang hanya menunjuk ke papernya). Paper yang ditemukan di hop >1 tetap
        diblokir supaya konteks tidak meledak lewat hub antar-paper.
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

        # Jatah budget PER HOP. Tanpa ini, node hub (paper dgn puluhan relasi langsung)
        # menghabiskan seluruh `max_rels` di hop-1, sehingga fakta hop-2 seperti
        # Paper -> Author -> Institution tidak pernah terekam.
        # max_rels=None: rekam semua relasi; pemangkasan diserahkan ke GraphFactReranker (menurut relevansi).
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
                        # Paper-blocking, dengan satu pengecualian: paper SUMBER (hop 1 dari
                        # node awal non-Paper) boleh diteruskan agar fakta tingkat-paper terjangkau.
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
        Susun hasil traversal jadi teks untuk LLM (roadmap 3.6).

        Format lama berupa daftar datar yang mengulang nama entitas di SETIAP baris
        (~40 char × 150 baris terbuang) dan mencampur semua tipe relasi, sehingga LLM sulit
        menemukan fakta spesifik. Format ini:
          - mengelompokkan relasi hop-1 PER TIPE RELASI (uses_dataset, finds, ...)
          - membuang pengulangan nama entitas
          - menyajikan relasi hop-2 sebagai RANTAI eksplisit (A -> relasi -> B) supaya
            penalaran multi-hop seperti Paper->Author->Institution terlihat jelas
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
                # Nama pendek digabung satu baris; item panjang (Finding/Limitation) per baris
                if all(len(i) <= 55 for i in items):
                    out.append(f"▸ {relation} ({len(items)}): " + " · ".join(items))
                else:
                    out.append(f"▸ {relation} ({len(items)}):")
                    out.extend(f"   • {i}" for i in items)

        if indirect:
            out.append("\n--- RELASI TIDAK LANGSUNG (rantai 2 hop) ---")
            out.extend(f"▸ {c}" for c in indirect)

        return "\n".join(out)
