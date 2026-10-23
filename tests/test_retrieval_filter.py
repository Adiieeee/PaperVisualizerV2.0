"""Ambang relevansi jalur vektor (fallback chunk + thematic). Tanpa model/jaringan: hasil Chroma dipalsukan."""
import types

import pytest

from src.query.resolver import QueryResolver, l2_to_cosine, relevant_hits


def _results(dists, docs=None, metas=None):
    docs = docs or [f"doc{i}" for i in range(len(dists))]
    return {"documents": [docs], "distances": [dists], "metadatas": [metas] if metas is not None else None}


@pytest.mark.parametrize("cos", [1.0, 0.75, 0.5, 0.0])
def test_l2_to_cosine_inverts_squared_l2_of_unit_vectors(cos):
    assert l2_to_cosine(2 - 2 * cos) == pytest.approx(cos)


def test_hits_below_threshold_are_dropped_and_order_kept():
    # cosine 0.80, 0.40, 0.60
    hits = relevant_hits(_results([0.4, 1.2, 0.8], metas=[{"doc_id": "a.pdf"}, {"doc_id": "b.pdf"}, {"doc_id": "c.pdf"}]), 0.5)
    assert [h["text"] for h in hits] == ["doc0", "doc2"]
    assert [h["doc_id"] for h in hits] == ["a.pdf", "c.pdf"]
    assert hits[0]["score"] == pytest.approx(0.8)


def test_all_below_threshold_gives_empty_list():
    assert relevant_hits(_results([1.3, 1.4]), 0.5) == []


@pytest.mark.parametrize("results", [{}, {"documents": [[]], "distances": [[]]}, {"documents": None}])
def test_empty_or_missing_results(results):
    assert relevant_hits(results, 0.5) == []


def test_missing_metadata_gives_no_doc_id():
    hits = relevant_hits(_results([0.2], metas=[None]), 0.5)
    assert hits == [{"text": "doc0", "doc_id": None, "score": pytest.approx(0.9)}]


def test_resolver_applies_configured_thresholds(monkeypatch):
    import src.query.resolver as R
    monkeypatch.setattr(R, "CHUNK_MIN_SIMILARITY", 0.7)
    monkeypatch.setattr(R, "COMMUNITY_MIN_SIMILARITY", 0.3)
    chroma = types.SimpleNamespace(
        search_chunks=lambda emb, n_results: _results([0.4, 0.8], metas=[{"doc_id": "a.pdf"}, {"doc_id": "b.pdf"}]),
        search_communities=lambda emb, n_results: _results([0.4, 1.2]),
    )
    r = object.__new__(QueryResolver)  # lewati __init__ (tanpa graph/model)
    r.embed = types.SimpleNamespace(embed_text=lambda q: [0.0])
    r.chroma = chroma
    assert [h["doc_id"] for h in r.search_chunks("q")] == ["a.pdf"]          # 0.8 lolos, 0.6 < 0.7
    assert r.search_communities("q") == ["doc0", "doc1"]                     # 0.8 dan 0.4 >= 0.3
