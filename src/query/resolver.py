import networkx as nx
from typing import List, Optional
import difflib
import re

from src.llm.embedding import EmbeddingClient
from src.vectorstore.chromadb_store import ChromaStore
from src.config import (
    CHUNK_MIN_SIMILARITY, COMMUNITY_MIN_SIMILARITY, RESOLVER_SIM_THRESHOLD, TOPIC_TERM_MIN_PAPERS,
    TOPIC_TERM_MIN_SHARE, load_model_to_paper, load_paper_metadata,
)

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
    """`term` appears whole in `text`. A hyphen counts as part of a word, so the node "RLHF" does NOT match
    "hh-rlhf" and "GPT" does not match "GPT-4o" (previously "-" was treated as a word boundary)."""
    return re.search(_TERM_LEFT + re.escape(term) + _TERM_RIGHT, text) is not None


_TERM_LEFT, _TERM_RIGHT = r"(?<![\w-])", r"(?![\w-])"

# Generic words in paper titles that distinguish nothing — dropped when computing
# title overlap (roadmap 1.5).
_TITLE_STOP = {
    "the", "a", "an", "of", "for", "in", "on", "and", "or", "to", "with", "from",
    "is", "are", "its", "not", "but", "how", "what", "as", "at", "by", "via",
}


def corpus_topic_terms(titles, min_share: float = TOPIC_TERM_MIN_SHARE,
                       min_papers: int = TOPIC_TERM_MIN_PAPERS) -> frozenset:
    """Title words (> 3 letters, not stop words) found in >= min_share of the titles: the corpus topic,
    e.g. "sycophancy" in 7 of 8 sycophancy papers. Empty when there are fewer than min_papers titles."""
    titles = [t for t in titles if t]
    if len(titles) < min_papers:
        return frozenset()
    counts = {}
    for t in titles:
        for w in {w for w in _WORD_RE.findall(t.lower()) if len(w) > 3 and w not in _TITLE_STOP}:
            counts[w] = counts.get(w, 0) + 1
    return frozenset(w for w, n in counts.items() if n >= min_share * len(titles))


def is_topic_only(name: str, topic_terms) -> bool:
    """True when a node name holds nothing but topic terms, short words and filler ("Sycophancy",
    "AI Sycophancy"); "Social Sycophancy" and "Sycophancy Rate" keep a content word, so they are real
    entities. Always False without topic terms."""
    if not topic_terms:
        return False
    words = _WORD_RE.findall(name.lower())
    if not any(w in topic_terms for w in words):
        return False
    return not [w for w in words if len(w) > 2 and w not in topic_terms and w not in _STOPWORDS]


class QueryResolver:
    # Defaults so a resolver built without __init__ (tests) behaves as if the corpus had no topic terms.
    topic_terms = frozenset()
    _topic_only = frozenset()

    def __init__(self, embed_client: EmbeddingClient, chroma_store: ChromaStore, graph: nx.Graph):
        self.embed = embed_client
        self.chroma = chroma_store
        self.graph = graph
        self._node_names = None   # lazy caches for embedding-based resolution
        self._node_embs = None
        self.model_to_paper = load_model_to_paper()  # auto-generated paper aliases (B2)
        # Corpus topic guard: nodes named only by the topic ("Sycophancy") are never resolved, because the
        # topic word is in nearly every question and pulled them into unrelated broad questions.
        self.topic_terms = corpus_topic_terms(m.get("title") for m in load_paper_metadata().values())
        self._topic_only = frozenset(n for n in graph.nodes() if is_topic_only(n, self.topic_terms))

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
            if node.lower() == ql and node not in self._topic_only:
                return node

        # 1.5. Partial Paper-title match. Users rarely mention a long title in full
        # ("paper Towards Understanding Sycophancy" vs the real 6-word title), so layer 2
        # below would instead pick a short generic node that happens to be contained ("Sycophancy").
        # STRICTLY GATED: at least 2 matching title words AND a ratio >= the threshold, so ordinary queries
        # (e.g. "metrik sycophancy apa saja?") are not hijacked to a Paper node.
        paper_hit = self._match_paper_title(ql, nodes)
        if paper_hit:
            return paper_hit

        # 2. Node name found verbatim within the query (whole-word); longest match wins.
        best_contained = None
        for node in nodes:
            nl = node.lower()
            if len(nl) < 3 or node in self._topic_only:
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
            if node in self._topic_only:
                continue
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
        Roadmap 1.2 — return ALL entities mentioned in the query (not just one),
        for comparison questions like "institusi penulis paper SycEval DAN ELEPHANT".

        Method: take every node name contained verbatim, then pick greedily from the
        LONGEST while dropping any whose **word span overlaps** — so "ResNet"
        is not also chosen once "ResNet-50" has been chosen. If the query mentions only one
        entity, fall back to the plain `resolve_entity()` so the old behavior is unchanged.
        """
        ctx = (context_query or query).lower()
        ql = query.strip().strip("?.!,").lower()
        paper_mode = bool(self.model_to_paper) and any(w in ctx for w in _PAPER_CONTEXT)

        spans = []
        for node in self.graph.nodes():
            nl = node.lower()
            if len(nl) < 3 or node in self._topic_only:
                continue
            for m in re.finditer(_TERM_LEFT + re.escape(nl) + _TERM_RIGHT, ql):
                spans.append((len(nl), m.start(), m.end(), node))
        spans.sort(reverse=True)  # longest wins

        chosen, taken = [], []
        for _, s, e, node in spans:
            if any(s < te and e > ts for ts, te in taken):
                continue  # overlaps with one already chosen
            taken.append((s, e))
            # If the query asks about paper attributes, map the entity name -> its Paper node
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
        Match the query to a Paper node by title-word overlap (roadmap 1.5).
        Two gates so it is not aggressive:
          - `min_words`: at least 2 meaningful title words must appear in the query (prevents a coincidental
            single word such as "sycophancy" from winning a random paper)
          - `min_ratio`: at least this fraction of the title words must appear in the query
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
            probe = self.embedding_probe(query)
            if probe is None:
                return None
            if self._node_embs is None:
                self._node_names = self.embedding_candidates()
                self._node_embs = self.embed.embed_batch(self._node_names)
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

    def embedding_probe(self, query: str) -> Optional[str]:
        """Layer-4 probe: the query without filler words, so the entity dominates its embedding. Topic terms
        are stripped too; when nothing but the topic is left, None (nothing specific to match)."""
        q = query.strip().strip("?.!,")
        tokens = [w for w in _WORD_RE.findall(q.lower()) if w not in _STOPWORDS]
        content = [w for w in tokens if w not in self.topic_terms]
        if tokens and not content:
            return None
        return " ".join(content) if content else q

    def embedding_candidates(self) -> List[str]:
        """Nodes layer 4 may return (topic-only nodes excluded)."""
        return [n for n in self.graph.nodes() if n not in self._topic_only]

    def mentions_topic(self, query: str) -> bool:
        """True when the query contains a corpus topic term (e.g. "sycophancy")."""
        ql = query.lower()
        return any(_contains_term(t, ql) for t in self.topic_terms)

    def search_chunks(self, query: str, top_k: int = 5) -> List[dict]:
        """Vector search on raw chunks as a fallback: [{text, doc_id, score}] above CHUNK_MIN_SIMILARITY."""
        emb = self.embed.embed_text(query)
        return relevant_hits(self.chroma.search_chunks(emb, n_results=top_k), CHUNK_MIN_SIMILARITY)

    def search_communities(self, query: str, top_k: int = 3) -> List[str]:
        """Vector search on community summaries for thematic queries (above COMMUNITY_MIN_SIMILARITY)."""
        emb = self.embed.embed_text(query)
        hits = relevant_hits(self.chroma.search_communities(emb, n_results=top_k), COMMUNITY_MIN_SIMILARITY)
        return [h["text"] for h in hits]


def l2_to_cosine(distance: float) -> float:
    """The Chroma collection is created without `hnsw:space`, so the distance is squared L2. For normalized vectors
    (bge-m3, normalize_embeddings=True) d = 2 - 2*cos, hence cos = 1 - d/2."""
    return 1.0 - distance / 2.0


def relevant_hits(results: dict, min_similarity: float) -> List[dict]:
    """Chroma query result -> [{text, doc_id, score}] with cosine >= min_similarity (Chroma's order is kept).
    ChromaDB always returns top-k, even when all of them are far from the question."""
    docs = (results.get("documents") or [[]])[0] or []
    dists = (results.get("distances") or [[]])[0] or []
    metas = (results.get("metadatas") or [[]])[0] or []
    hits = []
    for i, (text, dist) in enumerate(zip(docs, dists)):
        score = l2_to_cosine(dist)
        if score >= min_similarity:
            meta = metas[i] if i < len(metas) and metas[i] else {}
            hits.append({"text": text, "doc_id": meta.get("doc_id"), "score": score})
    return hits
