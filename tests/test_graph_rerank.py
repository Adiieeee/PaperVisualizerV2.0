"""GraphFactReranker: prune & order graph facts by relevance. No model: embeddings are faked."""
import networkx as nx
import pytest

from src.query.graph_rerank import GraphFactReranker, fact_text
from src.query.traversal import GraphTraversal


class _Embed:
    """1-dimensional vector: fact score = weight of the first keyword found in its text (query = [1.0])."""
    def __init__(self, weights):
        self.weights = weights
        self.batches = []

    def _vec(self, text):
        return [next((w for k, w in self.weights.items() if k in text), 0.0)]

    def embed_batch(self, texts):
        self.batches.append(list(texts))
        return [self._vec(t) for t in texts]

    def embed_text(self, text):
        return [1.0]


def _rel(source, target, relation, hop, description=None):
    return {"source": source, "target": target, "relation": relation, "hop": hop, "description": description}


def _trav(rels, entity="E"):
    return {"entity": entity, "type": "Method", "description": None, "relationships": rels}


def _pairs(result):
    return [(r["source"], r["target"]) for r in result["relationships"]]


def test_fact_text_includes_description_when_present():
    assert fact_text(_rel("A", "B", "uses_metric", 1)) == "A uses_metric B"
    assert fact_text(_rel("A", "B", "finds", 1, "x")) == "A finds B — x"
    assert fact_text(_rel("A", "B", None, 1)) == "A related_to B"


def test_under_budget_keeps_every_fact_and_orders_by_score():
    rels = [
        _rel("E", "P", "mentioned_in", 1),
        _rel("E", "M", "uses_metric", 1),
        _rel("P", "low", "finds", 2),
        _rel("P", "high", "finds", 2),
    ]
    rr = GraphFactReranker(_Embed({"uses_metric": 0.9, "high": 0.8, "low": 0.1, "mentioned_in": 0.2}))
    out = rr.select(_trav(rels), "q", 150)
    assert sorted(_pairs(out)) == sorted((r["source"], r["target"]) for r in rels)
    assert _pairs(out) == [("E", "M"), ("E", "P"), ("P", "high"), ("P", "low")]


def test_hop1_groups_are_kept_whole_and_ordered_by_best_member():
    rels = [_rel("E", f"d{i}", "uses_dataset", 1) for i in range(3)] + [_rel("E", "f", "finds", 1)]
    rr = GraphFactReranker(_Embed({"d0": 0.3, "d1": 0.2, "d2": 0.1, "finds": 0.9}))
    out = rr.select(_trav(rels), "q", 150)
    assert [r["relation"] for r in out["relationships"]] == ["finds", "uses_dataset", "uses_dataset", "uses_dataset"]


def test_hop1_share_is_capped_and_lowest_groups_dropped_first():
    # budget 5 -> hop-1 allowance = int(5 * 0.6) = 3
    rels = [_rel("E", "a1", "A", 1), _rel("E", "a2", "A", 1), _rel("E", "b1", "B", 1), _rel("E", "c1", "C", 1)]
    rr = GraphFactReranker(_Embed({"a1": 0.9, "a2": 0.8, "b1": 0.1, "c1": 0.5}))
    out = rr.select(_trav(rels), "q", 5)
    assert _pairs(out) == [("E", "a1"), ("E", "a2"), ("E", "c1")]


def test_hop2_is_pruned_by_score_and_never_orphaned():
    # budget 4 -> hop-1 cap 2. Hop-1 X is dropped, so the chain X -> * disappears too even though its score is high.
    rels = [
        _rel("E", "K", "uses_method", 1),
        _rel("E", "L", "uses_metric", 1),
        _rel("E", "X", "related_to", 1),
        _rel("K", "k_low", "finds", 2),
        _rel("K", "k_high", "finds", 2),
        _rel("L", "l_mid", "finds", 2),
        _rel("X", "x_top", "finds", 2),
    ]
    rr = GraphFactReranker(_Embed({"x_top": 0.99, "k_high": 0.7, "uses_method": 0.6, "uses_metric": 0.5,
                                   "l_mid": 0.4, "k_low": 0.1, "related_to": 0.05}))
    out = rr.select(_trav(rels), "q", 4)
    assert _pairs(out) == [("E", "K"), ("E", "L"), ("K", "k_high"), ("L", "l_mid")]


def test_error_or_empty_traversal_is_returned_unchanged():
    rr = GraphFactReranker(_Embed({}))
    err = {"error": "Node 'x' not found in the graph."}
    assert rr.select(err, "q", 150) is err
    empty = _trav([])
    assert rr.select(empty, "q", 150) is empty


def test_fact_embeddings_are_cached_across_queries():
    emb = _Embed({"M": 0.5})
    rr = GraphFactReranker(emb)
    trav = _trav([_rel("E", "M", "uses_metric", 1)])
    rr.select(trav, "q1", 150)
    rr.select(trav, "q2", 150)
    assert emb.batches == [["E uses_metric M"]]


def test_with_real_traversal_budget_is_respected():
    G = nx.Graph()
    G.add_node("E", type="Method")
    for i in range(10):
        G.add_node(f"n{i}", type="Method")
        G.add_edge("E", f"n{i}", relation="uses_method")
        for j in range(5):
            G.add_node(f"n{i}_{j}", type="Finding")
            G.add_edge(f"n{i}", f"n{i}_{j}", relation="finds")
    full = GraphTraversal(G).traverse("E", max_hops=2, max_rels=None)
    assert len(full["relationships"]) == 60
    out = GraphFactReranker(_Embed({})).select(full, "q", 20)
    rels = out["relationships"]
    assert len(rels) == 20 and sum(r["hop"] == 1 for r in rels) == 10
    kept_hop1 = {r["target"] for r in rels if r["hop"] == 1}
    assert all(r["source"] in kept_hop1 for r in rels if r["hop"] == 2)


@pytest.mark.parametrize("max_rels", [None, 150])
def test_traverse_unlimited_records_at_least_as_much(max_rels):
    G = nx.path_graph(5)
    nx.set_node_attributes(G, "Method", "type")
    nx.set_edge_attributes(G, "uses_method", "relation")
    res = GraphTraversal(G).traverse(0, max_hops=2, max_rels=max_rels)
    assert len(res["relationships"]) == 2


def test_fact_embeddings_persist_to_disk_and_are_reused(tmp_path):
    path = str(tmp_path / "fact_embeddings.npz")
    trav = _trav([_rel("E", "M", "uses_metric", 1)])
    first = _Embed({"M": 0.5})
    GraphFactReranker(first, cache_path=path).select(trav, "q", 150)
    assert first.batches == [["E uses_metric M"]]

    second = _Embed({"M": 0.5})  # "restart": a new instance, the cache is loaded from disk
    out = GraphFactReranker(second, cache_path=path).select(trav, "q", 150)
    assert second.batches == [] and _pairs(out) == [("E", "M")]


def test_cache_from_another_model_or_corrupt_file_is_ignored(tmp_path, monkeypatch):
    import src.query.graph_rerank as GR
    path = str(tmp_path / "fact_embeddings.npz")
    trav = _trav([_rel("E", "M", "uses_metric", 1)])
    GraphFactReranker(_Embed({}), cache_path=path).select(trav, "q", 150)

    monkeypatch.setattr(GR, "_CACHE_MODEL_ID", "other-model@rev/seq512")
    other = _Embed({})
    GraphFactReranker(other, cache_path=path).select(trav, "q", 150)
    assert other.batches == [["E uses_metric M"]]  # recomputed, not using another model's vectors

    with open(path, "wb") as f:
        f.write(b"bukan npz")
    broken = _Embed({})
    GraphFactReranker(broken, cache_path=path).select(trav, "q", 150)
    assert broken.batches == [["E uses_metric M"]]


def _small_graph():
    G = nx.Graph()
    G.add_node("P", type="Paper")
    G.add_node("M", type="Method")
    G.add_node("D", type="Dataset")
    G.add_node("F", type="Finding")
    G.add_edge("P", "M", relation="proposes_method")
    G.add_edge("M", "D", relation="evaluated_on", description="on D")
    G.add_edge("P", "F", relation="finds")
    return G


def test_all_fact_texts_cover_every_traversal_text():
    from src.query.graph_rerank import all_fact_texts
    G = _small_graph()
    texts = set(all_fact_texts(G))
    seen = {fact_text(r) for n in G for r in GraphTraversal(G).traverse(n, 2, None)["relationships"]}
    assert seen <= texts and len(texts) == 2 * G.number_of_edges()


def test_warm_embeds_only_missing_and_prunes_stale(tmp_path):
    path = str(tmp_path / "fact_embeddings.npz")
    G = _small_graph()
    emb = _Embed({})
    stats = GraphFactReranker(emb, cache_path=path).warm(G)
    assert stats == {"facts": 6, "computed": 6, "pruned": 0} and len(emb.batches) == 1

    G.remove_edge("P", "F")
    G.add_edge("D", "F", relation="related_to")
    emb2 = _Embed({})
    stats = GraphFactReranker(emb2, cache_path=path).warm(G)
    assert stats == {"facts": 6, "computed": 2, "pruned": 2}
    assert sorted(emb2.batches[0]) == ["D related_to F", "F related_to D"]

    emb3 = _Embed({})  # after warm, queries no longer embed facts
    GraphFactReranker(emb3, cache_path=path).select(GraphTraversal(G).traverse("M", 2, None), "q", 150)
    assert emb3.batches == []


def test_warm_on_an_empty_graph_writes_an_empty_cache(tmp_path):
    from src.query.graph_rerank import load_fact_cache
    path = str(tmp_path / "fact_embeddings.npz")
    GraphFactReranker(_Embed({}), cache_path=path).warm(_small_graph())
    stats = GraphFactReranker(_Embed({}), cache_path=path).warm(nx.Graph())
    assert stats == {"facts": 0, "computed": 0, "pruned": 6}
    assert load_fact_cache(path) == {}
