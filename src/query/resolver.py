import networkx as nx
from typing import List, Optional
import difflib
import re

from src.llm.embedding import EmbeddingClient
from src.vectorstore.chromadb_store import ChromaStore
from src.config import load_model_to_paper

# Query words that signal the user wants the PAPER (not the model node): authors, affiliation,
# findings, etc. In that case a model name like "VGG" should resolve to its Paper node.
_PAPER_CONTEXT = {
    "paper", "makalah", "penulis", "author", "authors", "ditulis", "wrote", "written",
    "afiliasi", "affiliation", "institusi", "institution", "temuan", "finding", "findings",
    "kesimpulan", "publikasi",
}

# Filler / category words stripped from a query before the semantic step, so a router
# phrase like "arsitektur yang dibandingkan dengan VGG" collapses toward the real entity.
_STOPWORDS = {
    "the", "a", "an", "of", "for", "with", "and", "or", "to", "in", "on", "by",
    "what", "which", "who", "does", "do", "is", "are", "use", "uses", "used", "about",
    "apa", "saja", "yang", "dengan", "adalah", "dari", "oleh", "dan", "atau", "itu",
    "paper", "papers", "makalah", "method", "methods", "metode", "dataset",
    "datasets", "metric", "metrics", "metrik", "author", "authors", "penulis",
    "architecture", "architectures", "arsitektur", "model", "models",
    "technique", "teknik", "network", "networks", "jaringan",
    "compared", "dibandingkan", "comparison", "perbandingan",
}

_WORD_RE = re.compile(r"\w+", re.UNICODE)


class QueryResolver:
    def __init__(self, embed_client: EmbeddingClient, chroma_store: ChromaStore, graph: nx.Graph):
        self.embed = embed_client
        self.chroma = chroma_store
        self.graph = graph
        self._node_names = None   # lazy caches for embedding-based resolution
        self._node_embs = None
        self.model_to_paper = load_model_to_paper()  # auto-generated paper aliases (B2)

    def resolve_entity(self, query: str, sim_threshold: float = 0.72,
                       context_query: str = None) -> Optional[str]:
        """
        Layered entity resolution → canonical graph node name (or None if truly unresolvable,
        in which case the caller falls back to chunk search):
          0. Paper-context (B2): if the query is about a paper's authors/affiliation/findings and
             mentions a popular model name, resolve to that model's PAPER node (models don't link
             to their own paper in text, so we use the PAPER_METADATA map).
          1. Exact match (case-insensitive)
          2. A node name appears verbatim inside the query (longest / most specific wins) —
             fixes the common case where the router returns a phrase, not just the entity
          3. Fuzzy string match (SequenceMatcher > 0.8)
          4. Vector-native: nearest graph node by LOCAL embedding similarity (semantic;
             lets "residual network" reach the "ResNet" node). Uses the vector layer to
             *enter* the graph rather than replace it.
        """
        nodes = list(self.graph.nodes())
        q = query.strip().strip("?.!,")
        ql = q.lower()

        # 0. Paper-context: map a popular model name -> its Paper node.
        ctx = (context_query or query).lower()
        if self.model_to_paper and any(w in ctx for w in _PAPER_CONTEXT):
            for model_lower, paper_title in self.model_to_paper.items():
                if re.search(r"(?<!\w)" + re.escape(model_lower) + r"(?!\w)", ctx):
                    for node in nodes:
                        if node.lower() == paper_title.lower():
                            print(f"Resolved (paper-context) '{query}' -> '{node}'")
                            return node

        # 1. Exact
        for node in nodes:
            if node.lower() == ql:
                return node

        # 2. Node name found verbatim within the query (whole-word); longest match wins.
        best_contained = None
        for node in nodes:
            nl = node.lower()
            if len(nl) < 3:
                continue
            if re.search(r"(?<!\w)" + re.escape(nl) + r"(?!\w)", ql):
                if best_contained is None or len(nl) > len(best_contained.lower()):
                    best_contained = node
        if best_contained:
            print(f"Resolved '{query}' -> '{best_contained}' (name found in query)")
            return best_contained

        # 3. Fuzzy whole-string
        best_match, highest = None, 0.0
        for node in nodes:
            r = difflib.SequenceMatcher(None, ql, node.lower()).ratio()
            if r > highest:
                highest, best_match = r, node
        if highest > 0.8:
            print(f"Fuzzy matched '{query}' -> '{best_match}' (ratio {highest:.2f})")
            return best_match

        # 4. Vector-native: nearest node by embedding
        node = self._embedding_resolve(q, sim_threshold)
        if node:
            return node

        print(f"Could not resolve entity '{query}' in the graph (falling back to chunk search).")
        return None

    def _embedding_resolve(self, query: str, threshold: float) -> Optional[str]:
        """Nearest graph node to the query by local embedding similarity (no API quota)."""
        try:
            if self._node_embs is None:
                self._node_names = list(self.graph.nodes())
                self._node_embs = self.embed.embed_batch(self._node_names)

            # Strip filler words so the entity dominates the query embedding.
            tokens = [w for w in _WORD_RE.findall(query.lower()) if w not in _STOPWORDS]
            probe = " ".join(tokens) if tokens else query
            qv = self.embed.embed_text(probe)

            best_i, best_sim = -1, -1.0
            for i, ev in enumerate(self._node_embs):
                s = self.embed.cosine_similarity(qv, ev)
                if s > best_sim:
                    best_sim, best_i = s, i
            if best_i >= 0 and best_sim >= threshold:
                node = self._node_names[best_i]
                print(f"Embedding matched '{query}' -> '{node}' (sim {best_sim:.2f})")
                return node
        except Exception as e:
            print(f"[Resolver] embedding resolution failed: {e}")
        return None

    def search_chunks(self, query: str, top_k: int = 5) -> List[str]:
        """Vector search on raw chunks as a fallback."""
        emb = self.embed.embed_text(query)
        results = self.chroma.search_chunks(emb, n_results=top_k)
        if not results['documents'] or not results['documents'][0]:
            return []
        return results['documents'][0]

    def search_communities(self, query: str, top_k: int = 3) -> List[str]:
        """Vector search on community summaries for thematic queries."""
        emb = self.embed.embed_text(query)
        results = self.chroma.search_communities(emb, n_results=top_k)
        if not results['documents'] or not results['documents'][0]:
            return []
        return results['documents'][0]
