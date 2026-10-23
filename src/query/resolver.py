import networkx as nx
from typing import List, Optional
import difflib
import re

from src.llm.embedding import EmbeddingClient
from src.vectorstore.chromadb_store import ChromaStore
from src.config import RESOLVER_SIM_THRESHOLD, load_model_to_paper

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


def _contains_term(term: str, text: str) -> bool:
    """`term` muncul utuh di `text`. Tanda hubung dihitung bagian kata, jadi node "RLHF" TIDAK cocok
    dengan "hh-rlhf" dan "GPT" tidak cocok dengan "GPT-4o" (sebelumnya "-" dianggap batas kata)."""
    return re.search(_TERM_LEFT + re.escape(term) + _TERM_RIGHT, text) is not None


_TERM_LEFT, _TERM_RIGHT = r"(?<![\w-])", r"(?![\w-])"

# Kata umum di judul paper yang tidak membedakan apa pun — dibuang saat menghitung
# tumpang-tindih judul (roadmap 1.5).
_TITLE_STOP = {
    "the", "a", "an", "of", "for", "in", "on", "and", "or", "to", "with", "from",
    "is", "are", "its", "not", "but", "how", "what", "as", "at", "by", "via",
}


class QueryResolver:
    def __init__(self, embed_client: EmbeddingClient, chroma_store: ChromaStore, graph: nx.Graph):
        self.embed = embed_client
        self.chroma = chroma_store
        self.graph = graph
        self._node_names = None   # lazy caches for embedding-based resolution
        self._node_embs = None
        self.model_to_paper = load_model_to_paper()  # auto-generated paper aliases (B2)

    def resolve_entity(self, query: str, sim_threshold: Optional[float] = None,
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
                if _contains_term(model_lower, ctx):
                    for node in nodes:
                        if node.lower() == paper_title.lower():
                            print(f"Resolved (paper-context) '{query}' -> '{node}'")
                            return node

        # 1. Exact
        for node in nodes:
            if node.lower() == ql:
                return node

        # 1.5. Judul Paper cocok sebagian. Judul panjang jarang disebut utuh oleh user
        # ("paper Towards Understanding Sycophancy" vs judul asli yang 6 kata), sehingga lapis 2
        # di bawah malah memenangkan node generik pendek yang kebetulan termuat ("Sycophancy").
        # DIGERBANGI KETAT: minimal 2 kata judul yang cocok DAN rasio >= ambang, supaya query
        # biasa (mis. "metrik sycophancy apa saja?") tidak terbajak ke node Paper.
        paper_hit = self._match_paper_title(ql, nodes)
        if paper_hit:
            return paper_hit

        # 2. Node name found verbatim within the query (whole-word); longest match wins.
        best_contained = None
        for node in nodes:
            nl = node.lower()
            if len(nl) < 3:
                continue
            if _contains_term(nl, ql):
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

        # 4. Vector-native: nearest node by embedding (threshold read at call time so it can be
        #    swept by scripts/calibrate_embedding.py)
        threshold = RESOLVER_SIM_THRESHOLD if sim_threshold is None else sim_threshold
        node = self._embedding_resolve(q, threshold)
        if node:
            return node

        print(f"Could not resolve entity '{query}' in the graph (falling back to chunk search).")
        return None

    def resolve_entities(self, query: str, context_query: str = None,
                         max_entities: int = 3) -> List[str]:
        """
        Roadmap 1.2 — kembalikan SEMUA entitas yang disebut di query (bukan hanya satu),
        untuk pertanyaan perbandingan seperti "institusi penulis paper SycEval DAN ELEPHANT".

        Cara: ambil semua nama node yang termuat verbatim, lalu pilih secara greedy dari yang
        TERPANJANG sambil membuang yang **rentang katanya tumpang tindih** — supaya "ResNet"
        tidak ikut terpilih saat "ResNet-50" sudah dipilih. Kalau query hanya menyebut satu
        entitas, jatuh kembali ke `resolve_entity()` biasa sehingga perilaku lama tak berubah.
        """
        ctx = (context_query or query).lower()
        ql = query.strip().strip("?.!,").lower()
        paper_mode = bool(self.model_to_paper) and any(w in ctx for w in _PAPER_CONTEXT)

        spans = []
        for node in self.graph.nodes():
            nl = node.lower()
            if len(nl) < 3:
                continue
            for m in re.finditer(_TERM_LEFT + re.escape(nl) + _TERM_RIGHT, ql):
                spans.append((len(nl), m.start(), m.end(), node))
        spans.sort(reverse=True)  # terpanjang menang

        chosen, taken = [], []
        for _, s, e, node in spans:
            if any(s < te and e > ts for ts, te in taken):
                continue  # tumpang tindih dgn yang sudah dipilih
            taken.append((s, e))
            # Kalau query bertanya soal atribut paper, petakan nama entitas -> node Paper-nya
            target = node
            if paper_mode:
                title = self.model_to_paper.get(node.lower())
                if title:
                    for n2 in self.graph.nodes():
                        if n2.lower() == title.lower():
                            target = n2
                            break
            if target not in chosen:
                chosen.append(target)
            if len(chosen) >= max_entities:
                break

        if len(chosen) >= 2:
            print(f"Resolved {len(chosen)} entitas (multi): {chosen}")
            return chosen

        single = self.resolve_entity(query, context_query=context_query)
        return [single] if single else []

    def _match_paper_title(self, query_lower: str, nodes, min_ratio: float = 0.55,
                           min_words: int = 2) -> Optional[str]:
        """
        Cocokkan query ke node Paper lewat tumpang-tindih kata judul (roadmap 1.5).
        Dua gerbang supaya tidak agresif:
          - `min_words`: minimal 2 kata judul bermakna harus muncul di query (cegah kebetulan
            satu kata seperti "sycophancy" memenangkan paper acak)
          - `min_ratio`: minimal sekian persen kata judul muncul di query
        """
        qwords = set(_WORD_RE.findall(query_lower))
        best, best_ratio = None, 0.0
        for n in nodes:
            if self.graph.nodes[n].get("type") != "Paper":
                continue
            twords = {w for w in _WORD_RE.findall(n.lower())
                      if w not in _TITLE_STOP and len(w) > 2}
            if not twords:
                continue
            matched = twords & qwords
            ratio = len(matched) / len(twords)
            if len(matched) >= min_words and ratio > best_ratio:
                best, best_ratio = n, ratio
        if best and best_ratio >= min_ratio:
            print(f"Resolved (judul paper cocok {best_ratio:.0%}) -> '{best[:50]}'")
            return best
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
