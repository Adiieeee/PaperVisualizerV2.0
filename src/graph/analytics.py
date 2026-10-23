import networkx as nx
from networkx.algorithms.community import louvain_communities
from collections import Counter, defaultdict
from itertools import combinations
from typing import List, Dict, Any

from src.config import COMMUNITY_EXCLUDED_TYPES, COMMUNITY_MIN_SIZE, COMMUNITY_SEED

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

    def _node_chunks(self, n) -> set:
        """Chunk tempat node disebut: atribut node + semua edge-nya (termasuk edge ke Paper,
        tempat sebagian besar source_chunk_id berada)."""
        chunks = {self.graph.nodes[n].get("source_chunk_id", "")}
        chunks.update(d.get("source_chunk_id", "") for _, _, d in self.graph.edges(n, data=True))
        return {c for c in chunks if c}

    def _community_graph(self) -> nx.Graph:
        """
        Graf ko-okurensi untuk deteksi community. Node: semua tipe kecuali
        COMMUNITY_EXCLUDED_TYPES. Bobot edge = jumlah chunk yang dibagi bersama, +1 untuk
        tiap relasi asli di antara keduanya (termasuk variant_of).
        """
        nodes = [n for n, d in self.graph.nodes(data=True)
                 if d.get("type") not in COMMUNITY_EXCLUDED_TYPES]
        H = nx.Graph()
        H.add_nodes_from(nodes)
        by_chunk = defaultdict(list)
        for n in nodes:
            for c in self._node_chunks(n):
                by_chunk[c].append(n)
        pairs = Counter()
        for members in by_chunk.values():
            pairs.update(combinations(sorted(members), 2))
        for u, v in self.graph.subgraph(nodes).edges():
            pairs[tuple(sorted((u, v)))] += 1
        for (u, v), w in pairs.items():
            H.add_edge(u, v, weight=float(w))
        return H

    def detect_communities(self, min_size: int = COMMUNITY_MIN_SIZE) -> List[Dict[str, Any]]:
        """
        Detects communities using weighted Louvain over the chunk co-occurrence graph.
        Paper nodes are excluded (they would act as hubs, giving one community per paper), and
        so are Author/Institution. Communities smaller than min_size are dropped: those nodes
        are outliers. Ids are 0..N-1 by size (desc), then name, so a rebuild is reproducible.
        Returns a list of dicts: [{'community_id': int, 'nodes': List[str]}]
        """
        H = self._community_graph()
        comms = louvain_communities(H, weight="weight", seed=COMMUNITY_SEED)
        comms = sorted((sorted(c) for c in comms if len(c) >= min_size), key=lambda c: (-len(c), c))
        return [{"community_id": i, "nodes": c} for i, c in enumerate(comms)]

    def community_context(self, nodes: List[str], max_nodes: int = 25, max_rels: int = 25) -> str:
        """
        Teks untuk prompt ringkasan community: anggota (tipe + deskripsi singkat), diurutkan
        menurut bobot keterhubungan di graf community, lalu relasi asli antar-anggota.
        Teksnya berasal dari dokumen tak tepercaya; summarize_community yang menetralkannya.
        """
        H = self._community_graph()
        members = set(nodes)
        ranked = sorted(nodes, key=lambda n: (-H.degree(n, weight="weight") if n in H else 0, n))
        lines = ["Entities:"]
        for n in ranked[:max_nodes]:
            d = self.graph.nodes[n]
            desc = (d.get("description") or "").strip()
            if len(desc) > 150:
                desc = desc[:150].rstrip() + "..."
            lines.append(f"- {n} ({d.get('type', '')})" + (f": {desc}" if desc else ""))
        if len(ranked) > max_nodes:
            lines.append(f"- ... and {len(ranked) - max_nodes} more")
        rels = sorted(
            (u, d.get("relation", ""), v) for u, v, d in self.graph.subgraph(members).edges(data=True)
            if d.get("relation") != "mentioned_in"
        )
        if rels:
            lines.append("Relations:")
            lines.extend(f"- {u} -[{r}]- {v}" for u, r, v in rels[:max_rels])
        return "\n".join(lines)
