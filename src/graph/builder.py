import re
import networkx as nx
from collections import Counter
from typing import List, Tuple
from src.config import ExtractedKnowledge
from src.security.integrity import atomic_write

class GraphBuilder:
    def __init__(self):
        self.graph = nx.Graph()

    @staticmethod
    def _doc_of(chunk_id) -> str:
        """Derive the document id from a chunk id ('{doc_id}_chunk_{n}')."""
        if not chunk_id:
            return ""
        return chunk_id.rsplit("_chunk_", 1)[0]

    @staticmethod
    def _paper_name_from_doc(doc_id: str) -> str:
        """Fallback Paper node name when a document yielded no Paper entity."""
        name = doc_id
        for ext in (".pdf", ".PDF"):
            if name.endswith(ext):
                name = name[: -len(ext)]
        return f"Paper {name}"

    def add_knowledge_with_provenance(self, knowledge_items: List[Tuple[str, ExtractedKnowledge]]):
        """
        Phase A: build the graph AND anchor every entity to the Paper node of the
        document it was extracted from, so nothing floats unconnected (DESIGN_REFERENCE §11).

        knowledge_items: list of (chunk_id, ExtractedKnowledge); doc_id is derived from
        chunk_id. This only shapes graph STRUCTURE — the query pipeline, dedup, and vector
        store are untouched. `mentioned_in` edges all touch a Paper node, which Louvain
        already excludes, so community detection is unaffected.
        """
        # canonical entity name -> set of doc_ids it was extracted from
        entity_docs = {}
        # doc_id -> Counter of Paper-entity names seen in that doc (to pick a title)
        doc_paper_candidates = {}

        # Pass 1: entity nodes + provenance accumulation
        for chunk_id, k in knowledge_items:
            doc_id = self._doc_of(chunk_id)
            for ent in k.entities:
                if not self.graph.has_node(ent.name):
                    self.graph.add_node(
                        ent.name,
                        type=ent.type or "",
                        description=ent.description or "",
                        source_chunk_id=chunk_id or "",
                    )
                elif not self.graph.nodes[ent.name].get("description") and ent.description:
                    self.graph.nodes[ent.name]["description"] = ent.description

                if doc_id:
                    entity_docs.setdefault(ent.name, set()).add(doc_id)
                    if ent.type == "Paper":
                        doc_paper_candidates.setdefault(doc_id, Counter())[ent.name] += 1

        # Pass 2: edges — only between known nodes, never self-loops (same guard as add_knowledge)
        skipped_dangling = 0
        skipped_selfloop = 0
        for chunk_id, k in knowledge_items:
            for rel in k.relationships:
                if rel.source_entity == rel.target_entity:
                    skipped_selfloop += 1
                    continue
                if not (self.graph.has_node(rel.source_entity) and self.graph.has_node(rel.target_entity)):
                    skipped_dangling += 1
                    continue
                if self.graph.has_edge(rel.source_entity, rel.target_entity):
                    continue
                self.graph.add_edge(
                    rel.source_entity,
                    rel.target_entity,
                    relation=rel.relation_type or "",
                    description=rel.description or "",
                    source_chunk_id=chunk_id or "",
                )
        if skipped_dangling or skipped_selfloop:
            print(f"[Builder] Skipped {skipped_dangling} dangling edges "
                  f"(endpoint not an extracted entity) and {skipped_selfloop} self-loops.")

        # Resolve one canonical Paper node per document.
        doc_paper = {}
        all_doc_ids = {d for docs in entity_docs.values() for d in docs}
        for doc_id in all_doc_ids:
            cands = doc_paper_candidates.get(doc_id)
            if cands:
                # most frequently seen title; tie-break by longest (more specific) name
                best = max(cands.items(), key=lambda kv: (kv[1], len(kv[0])))[0]
            else:
                # no Paper entity extracted for this doc — synthesize one from the filename
                best = self._paper_name_from_doc(doc_id)
                if not self.graph.has_node(best):
                    self.graph.add_node(best, type="Paper", description="", source_chunk_id="")
            doc_paper[doc_id] = best

        # Anchor every non-Paper node to its document's Paper node via `mentioned_in`.
        connected = 0
        for name, docs in entity_docs.items():
            if self.graph.nodes[name].get("type") == "Paper":
                continue
            papers = sorted({doc_paper[d] for d in docs if d in doc_paper})
            if papers and not self.graph.nodes[name].get("source_paper"):
                self.graph.nodes[name]["source_paper"] = "; ".join(papers)
            for d in docs:
                paper_node = doc_paper.get(d)
                if not paper_node or paper_node == name:
                    continue
                # only add if there is no existing path (avoids redundant anchors)
                if not nx.has_path(self.graph, name, paper_node):
                    self.graph.add_edge(
                        name, paper_node,
                        relation="mentioned_in",
                        description="Entity found in this paper",
                        source_chunk_id="",
                    )
                    connected += 1
        if connected:
            print(f"[Builder] Anchored {connected} orphan node(s) to their source Paper via mentioned_in.")

    def add_knowledge(self, knowledge_list: List[ExtractedKnowledge]):
        """
        Adds deduplicated entities and relationships to the NetworkX graph.
        GraphML does not support None values — all attributes are coerced to strings.

        Two passes: all entity nodes are added first, so an edge is kept as long as
        BOTH endpoints exist as entity nodes somewhere in the corpus. Edges whose
        endpoints were never extracted as entities are skipped instead of silently
        auto-creating typeless "phantom" nodes (NetworkX add_edge default behavior).
        Self-loops — which can appear once two entities merge to the same canonical
        name during dedup — are also skipped.
        """
        # Pass 1: entity nodes
        for k in knowledge_list:
            for ent in k.entities:
                if not self.graph.has_node(ent.name):
                    self.graph.add_node(
                        ent.name,
                        type=ent.type or "",
                        description=ent.description or "",
                        source_chunk_id=getattr(ent, "_source_chunk_id", "") or "",
                    )
                else:
                    # Update description if it's currently empty
                    if not self.graph.nodes[ent.name].get('description') and ent.description:
                        self.graph.nodes[ent.name]['description'] = ent.description

        # Pass 2: edges — only between known nodes, never self-loops
        skipped_dangling = 0
        skipped_selfloop = 0
        for k in knowledge_list:
            for rel in k.relationships:
                if rel.source_entity == rel.target_entity:
                    skipped_selfloop += 1
                    continue
                if not (self.graph.has_node(rel.source_entity) and self.graph.has_node(rel.target_entity)):
                    skipped_dangling += 1
                    continue
                if self.graph.has_edge(rel.source_entity, rel.target_entity):
                    continue
                self.graph.add_edge(
                    rel.source_entity,
                    rel.target_entity,
                    relation=rel.relation_type or "",
                    description=rel.description or "",
                    source_chunk_id=getattr(rel, "_source_chunk_id", "") or "",
                )

        if skipped_dangling or skipped_selfloop:
            print(f"[Builder] Skipped {skipped_dangling} dangling edges "
                  f"(endpoint not an extracted entity) and {skipped_selfloop} self-loops.")

    def link_models_to_papers(self, paper_metadata: dict):
        """
        B2: deterministically connect each paper's popular model name(s) to its Paper node via
        `proposes_method`. Papers don't use their own popular names in text, so this hand-map is
        the only reliable way to bridge e.g. the 'VGG' node to the 'Very Deep Convolutional…' paper.
        Ensures the canonical Paper node exists, then links any matching model node to it.
        """
        # case-insensitive lookup of existing node names (avoid creating a duplicate Paper node
        # when the extracted title differs only in case from the canonical one).
        lower_index = {n.lower(): n for n in self.graph.nodes()}
        linked = 0
        for meta in paper_metadata.values():
            title = meta["title"]
            paper_node = lower_index.get(title.lower())
            if paper_node is None:
                self.graph.add_node(title, type="Paper", description="", source_chunk_id="")
                paper_node = title
                lower_index[title.lower()] = title
            for m in meta["models"]:
                node = lower_index.get(m.lower())
                if node and node != paper_node and not self.graph.has_edge(paper_node, node):
                    self.graph.add_edge(
                        paper_node, node,
                        relation="proposes_method",
                        description=f"{paper_node} introduces {node}.",
                        source_chunk_id="",
                    )
                    linked += 1
        if linked:
            print(f"[Builder] Linked {linked} model node(s) to their source paper (B2).")

    # Nama varian bergaya "<huruf><angka versi>": ResNet-50, llama-2-70b-chat, CIFAR-100,
    # Gemini-1.5-Pro. Murni pola penulisan — tidak ada daftar nama yang di-hardcode, sehingga
    # bekerja untuk korpus/domain apa pun.
    _FAMILY_BASE_RE = re.compile(r"^([A-Za-z][A-Za-z\s]{2,}?)[\s\-_]*\d")

    @classmethod
    def _family_base(cls, name: str):
        m = cls._FAMILY_BASE_RE.match(name.strip())
        if not m:
            return None
        base = re.sub(r"\s+", " ", m.group(1)).strip()
        return base if len(base) >= 3 else None

    def link_entity_families(self, min_variants: int = 2, min_papers: int = 2):
        """
        Hubungkan varian bertingkat-versi ke satu node "keluarga" lewat relasi `variant_of`
        (mis. llama-2-70b-chat / LLaMA 3 / Llama-17B  ->  Llama).

        Tujuannya konektivitas, BUKAN penggabungan: setiap varian tetap node tersendiri, jadi
        guard CRITICAL 1 (ResNet-44 != ResNet-56) tidak dilanggar. Tanpa ini, entitas yang
        sebenarnya satu keluarga tersebar dan hanya menempel ke paper-nya masing-masing.

        Keluarga hanya dibuat kalau benar-benar menambah konektivitas: minimal `min_variants`
        varian DAN tersebar di minimal `min_papers` paper berbeda.
        """
        papers = {n for n, d in self.graph.nodes(data=True) if d.get("type") == "Paper"}
        lower_index = {n.lower(): n for n in self.graph.nodes()}

        groups = {}
        for n, d in self.graph.nodes(data=True):
            if n in papers:
                continue
            base = self._family_base(n)
            if base:
                groups.setdefault((d.get("type", ""), base.lower()), []).append((n, base))

        created = linked = 0
        for (ent_type, base_key), members in groups.items():
            if len(members) < min_variants:
                continue
            covered = set()
            for name, _ in members:
                covered |= {v for v in self.graph.neighbors(name) if v in papers}
            if len(covered) < min_papers:
                continue

            # Pakai node yang sudah ada bila namanya persis basis keluarga; kalau tidak,
            # buat node baru memakai kapitalisasi yang paling sering dipakai varian-variannya.
            family = lower_index.get(base_key)
            if family is None:
                spellings = Counter(b for _, b in members)
                family = spellings.most_common(1)[0][0]
                self.graph.add_node(family, type=ent_type, description="", source_chunk_id="")
                lower_index[base_key] = family
                created += 1

            for name, _ in members:
                if name != family and not self.graph.has_edge(name, family):
                    self.graph.add_edge(
                        name, family,
                        relation="variant_of",
                        description=f"{name} adalah varian dari {family}.",
                        source_chunk_id="",
                    )
                    linked += 1

        if linked:
            print(f"[Builder] Menautkan {linked} varian ke {created} node keluarga baru "
                  f"(relasi variant_of).")

    def family_nodes(self) -> dict:
        """{node_keluarga: [varian...]} — node yang jadi induk >=2 edge variant_of."""
        members = {}
        for u, v, d in self.graph.edges(data=True):
            if d.get("relation") != "variant_of":
                continue
            # arah tersimpan sebagai (varian, keluarga) saat dibuat, tapi GraphML undirected:
            # induk = node yang punya lebih banyak tautan variant_of.
            du = sum(1 for _, _, e in self.graph.edges(u, data=True) if e.get("relation") == "variant_of")
            dv = sum(1 for _, _, e in self.graph.edges(v, data=True) if e.get("relation") == "variant_of")
            parent, child = (u, v) if du >= dv else (v, u)
            members.setdefault(parent, []).append(child)
        return {k: v for k, v in members.items() if len(v) >= 2}

    def merge_family_aliases(self, llm):
        """
        Gabungkan keluarga yang sebenarnya sama tapi beda penyebutan (mis. ChatGPT vs GPT)
        dengan adjudikasi LLM. Kandidat dibatasi pada pasangan yang salah satu namanya termuat
        di nama lain — jadi jumlah panggilan LLM sangat sedikit (biasanya 0-3 per korpus).
        LLM dipakai karena ia tahu ChatGPT≈GPT tapi ResNet≠DenseNet; pencocokan string murni
        akan salah menggabungkan keduanya.
        """
        fams = self.family_nodes()
        names = sorted(fams)
        candidates = [
            (big, small) for big in names for small in names
            if big != small and small.lower() in big.lower()
        ]
        if not candidates:
            return

        merged = 0
        for big, small in candidates:
            if not (self.graph.has_node(big) and self.graph.has_node(small)):
                continue
            try:
                decision = llm.resolve_family_merge(big, fams.get(big, []), small, fams.get(small, []))
            except Exception as e:
                print(f"[Builder] Adjudikasi keluarga '{big}' vs '{small}' gagal: {e}")
                continue

            if decision.same_family:
                parent = small if decision.canonical.lower().strip() == small.lower() else big
                child = big if parent == small else small
                if parent != child and not self.graph.has_edge(child, parent):
                    self.graph.add_edge(
                        child, parent,
                        relation="variant_of",
                        description=f"{child} adalah penyebutan lain dari keluarga {parent}.",
                        source_chunk_id="",
                    )
                    merged += 1
                    print(f"  [Family] LLM: '{child}' dan '{parent}' keluarga sama -> ditautkan "
                          f"({decision.reason[:60]})")
            else:
                print(f"  [Family] LLM: '{big}' dan '{small}' BUKAN keluarga sama "
                      f"({decision.reason[:60]})")

        if merged:
            print(f"[Builder] {merged} keluarga digabungkan lewat adjudikasi LLM.")

    def reclassify_noncanonical_papers(self, paper_metadata: dict):
        """
        A merely-cited paper (e.g. "Network in Network") can leak in as a Paper node. Keep only the
        corpus's OWN source papers as type=Paper; reclassify the rest to Method (they are really
        architecture references). Uses the auto-generated paper_metadata titles — no hardcoding.
        """
        canonical = {m["title"].lower() for m in paper_metadata.values() if m.get("title")}
        if not canonical:
            return
        changed = 0
        for n, d in self.graph.nodes(data=True):
            if d.get("type") == "Paper" and n.lower() not in canonical:
                self.graph.nodes[n]["type"] = "Method"
                changed += 1
        if changed:
            print(f"[Builder] Reclassified {changed} non-source 'Paper' node(s) to Method (cited refs).")

    def save_graph(self, filepath: str):
        """
        Saves the graph to GraphML format (atomic: temp file + os.replace, so a crash mid-write
        can never leave a truncated graph.graphml).
        """
        atomic_write(filepath, lambda tmp: nx.write_graphml(self.graph, tmp))
        print(f"Graph saved to {filepath} with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")

    def load_graph(self, filepath: str):
        """
        Loads the graph from GraphML format.
        """
        self.graph = nx.read_graphml(filepath)
        print(f"Graph loaded from {filepath}.")
