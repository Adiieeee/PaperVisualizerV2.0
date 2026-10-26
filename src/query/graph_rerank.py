"""
Rerank graph facts against the question: prune nodes that exceed the budget by relevance (not by
BFS/GraphML order) and order the context so the most relevant facts appear first.

Score = bge-m3 cosine (the embedding model the engine has already loaded, no new model) between the user's
original question and the fact text "source relation target — description". Embedding facts is expensive
(bge-m3 568M parameters: ~25 ms/fact on a GTX 1650, ~80 ms on CPU; the GPT-4o node has 301 facts), so fact
embeddings are cached per text in memory AND on disk (FACT_EMBEDDINGS_PATH): each fact is computed once,
and restarts stay fast. The cache is addressed by text and only used when the model/revision/seq-length are
the same, so it is safe across corpora.

Rules (tested in tests/test_graph_rerank.py):
  - the hop-1 allowance stays <= 60% of the budget (same as traverse()), the rest goes to hop-2;
  - hop-1 is chosen per relation GROUP (group score = highest member score); groups are taken whole as long as
    they fit, so a list like `uses_dataset` is not cut partway for no reason;
  - hop-2 facts are kept only if their parent hop-1 edge is kept (the chain A → B → C does not break);
  - output order: hop-1 groups by group score, hop-2 by score. format_traversal_context
    preserves list order, so the format and the answer prompt do not change. A node under the budget
    loses no facts; only their order changes.
"""
import os
from typing import Any, Dict, List, Optional

import numpy as np

from src.config import EMBEDDING_MAX_SEQ_LENGTH, EMBEDDING_MODEL_NAME, EMBEDDING_MODEL_REVISION
from src.security.integrity import atomic_write

HOP1_SHARE = 0.6  # same as hop1_cap in GraphTraversal.traverse
_CACHE_MODEL_ID = f"{EMBEDDING_MODEL_NAME}@{EMBEDDING_MODEL_REVISION}/seq{EMBEDDING_MAX_SEQ_LENGTH}"


def fact_text(rel: Dict[str, Any]) -> str:
    text = f"{rel['source']} {rel.get('relation') or 'related_to'} {rel['target']}"
    return f"{text} — {rel['description']}" if rel.get("description") else text


def all_fact_texts(graph) -> List[str]:
    """All fact texts that can appear in the context: every edge from BOTH directions, because the source/target
    direction is decided by traversal (the graph is undirected). Attributes are read exactly as in traverse()."""
    texts = []
    for u, v, ed in graph.edges(data=True):
        for source, target in ((u, v), (v, u)):
            texts.append(fact_text({"source": source, "target": target,
                                    "relation": ed.get("relation"), "description": ed.get("description")}))
    return list(dict.fromkeys(texts))


def load_fact_cache(path: str) -> Dict[str, np.ndarray]:
    """{fact text: vector} from disk; {} if missing, corrupt, or made by another model."""
    if not path or not os.path.exists(path):
        return {}
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(data["model"]) != _CACHE_MODEL_ID:
                return {}
            return {str(t): v for t, v in zip(data["texts"], data["vectors"].astype(np.float32))}
    except Exception as e:  # corrupt file: just recompute, don't fail the query
        print(f"[Rerank] Cache embedding fakta diabaikan ({e})")
        return {}


def save_fact_cache(path: str, cache: Dict[str, np.ndarray]) -> None:
    texts = list(cache)

    def _write(tmp: str) -> None:
        with open(tmp, "wb") as f:  # file object: np.savez appends ".npz" to a file name without that extension
            vectors = np.stack([cache[t] for t in texts]) if texts else np.zeros((0, 0), dtype=np.float32)
            np.savez(f, model=np.array(_CACHE_MODEL_ID), texts=np.array(texts, dtype=str), vectors=vectors)

    atomic_write(path, _write)


class GraphFactReranker:
    def __init__(self, embed_client, cache_path: Optional[str] = None):
        self.embed = embed_client
        self.cache_path = cache_path
        self._cache: Optional[Dict[str, np.ndarray]] = None  # loaded from disk on first use

    def _loaded(self) -> Dict[str, np.ndarray]:
        if self._cache is None:
            self._cache = load_fact_cache(self.cache_path)
        return self._cache

    def _save(self) -> None:
        if self.cache_path:
            try:
                save_fact_cache(self.cache_path, self._cache)
            except OSError as e:  # disk full / read-only: keep going with the in-memory cache
                print(f"[Rerank] Cache embedding fakta tidak tersimpan ({e})")

    def _embed_missing(self, texts: List[str]) -> int:
        cache = self._loaded()
        missing = [t for t in dict.fromkeys(texts) if t not in cache]
        if missing:
            for t, vec in zip(missing, self.embed.embed_batch(missing)):
                cache[t] = np.asarray(vec, dtype=np.float32)
            self._save()
        return len(missing)

    def warm(self, graph) -> Dict[str, int]:
        """Fill the cache for ALL graph facts (called at the end of the build) and drop texts no longer in the
        graph. Only facts not yet in the cache are embedded, so rebuilding a similar corpus is fast."""
        texts = all_fact_texts(graph)
        cache = self._loaded()
        wanted = set(texts)
        stale = [t for t in cache if t not in wanted]
        for t in stale:
            del cache[t]
        computed = self._embed_missing(texts)
        if stale and not computed:
            self._save()
        return {"facts": len(texts), "computed": computed, "pruned": len(stale)}

    def score(self, query: str, rels: List[Dict[str, Any]]) -> List[float]:
        texts = [fact_text(r) for r in rels]
        self._embed_missing(texts)
        q = np.asarray(self.embed.embed_text(query), dtype=np.float32)
        return [float(np.dot(q, self._cache[t])) for t in texts]

    def select(self, trav: Dict[str, Any], query: str, max_rels: int) -> Dict[str, Any]:
        """Full traverse() result -> a copy with <= max_rels relations, ordered by relevance."""
        rels = trav.get("relationships") or []
        if "error" in trav or not rels:
            return trav
        scores = self.score(query, rels)
        entity = trav["entity"]

        groups: Dict[str, List[int]] = {}
        hop2: List[int] = []
        for i, r in enumerate(rels):
            if r["hop"] == 1 and r["source"] == entity:
                groups.setdefault(r.get("relation") or "", []).append(i)
            else:
                hop2.append(i)

        by_score = lambda idx: sorted(idx, key=lambda i: -scores[i])  # sorted() is stable for equal scores
        hop1_cap = max(1, int(max_rels * HOP1_SHARE))
        kept: List[int] = []
        for members in sorted(groups.values(), key=lambda m: -max(scores[i] for i in m)):
            room = hop1_cap - len(kept)
            if room <= 0:
                break
            kept.extend(by_score(members)[:room])

        reachable = {rels[i]["target"] for i in kept}
        room = max_rels - len(kept)
        kept.extend(by_score([i for i in hop2 if rels[i]["source"] in reachable])[:max(room, 0)])

        return {**trav, "relationships": [rels[i] for i in kept]}
