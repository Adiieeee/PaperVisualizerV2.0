"""
GraphRAGEngine.query must tag the context according to its ORIGIN (graph / raw text /
community summaries). Importing src.engine loads torch (~9 s); this file can be skipped with `-k "not engine"`.
"""
import types

import pytest

from src.config import QueryRouting
from src.engine import NO_CONTEXT_MESSAGE, GraphRAGEngine, format_chunk
from src.llm.prompts import KIND_COMMUNITY, KIND_GRAPH, KIND_TEXT


class _LLM:
    def __init__(self):
        self.calls = []

    def generate_final_answer(self, query, blocks):
        self.calls.append((query, blocks))
        return "jawaban"


def _engine(routing, **parts):
    e = object.__new__(GraphRAGEngine)  # skip __init__ (does not load the model/graph/Chroma)
    e.llm = _LLM()
    e.router = types.SimpleNamespace(route=lambda q: routing)
    e.resolver = parts.get("resolver")
    e.traversal = parts.get("traversal")
    e.analytics = parts.get("analytics")
    e.paper_titles = parts.get("paper_titles", {})
    e.reranker = parts.get("reranker", types.SimpleNamespace(select=lambda trav, q, max_rels: trav))
    return e


def _only_block(engine):
    (query, blocks), = engine.llm.calls
    (kind, text), = blocks
    return kind, text


def test_counting_is_graph_context():
    an = types.SimpleNamespace(get_usage_stats=lambda t: [("SFT", 2, 3), ("RLHF", 1, 1)])
    e = _engine(QueryRouting(intent="counting", target_type="Method"), analytics=an)
    assert e.query("metode terbanyak?") == "jawaban"
    kind, text = _only_block(e)
    assert kind == KIND_GRAPH and "SFT: 2 papers, 3 connections" in text and "number of papers" in text


def test_structural_is_graph_context():
    an = types.SimpleNamespace(get_bridge_entities=lambda: [("GPT-4o", ["P1", "P2"], 2)])
    e = _engine(QueryRouting(intent="structural"), analytics=an)
    e.query("apa yang jadi jembatan?")
    kind, text = _only_block(e)
    assert kind == KIND_GRAPH and "GPT-4o" in text


def test_thematic_is_community_context():
    rs = types.SimpleNamespace(search_communities=lambda q: ["ringkasan A"])
    e = _engine(QueryRouting(intent="thematic", search_query="tren"), resolver=rs)
    e.query("tren?")
    kind, text = _only_block(e)
    assert kind == KIND_COMMUNITY and "ringkasan A" in text


def test_entity_found_is_graph_context():
    rs = types.SimpleNamespace(resolve_entities=lambda q, context_query=None: ["SycEval"])
    tr = types.SimpleNamespace(
        traverse=lambda n, max_hops, max_rels: {"entity": n},
        format_traversal_context=lambda r: f"ENTITAS: {r['entity']}",
    )
    e = _engine(QueryRouting(intent="entity", search_query="SycEval"), resolver=rs, traversal=tr)
    e.query("dataset SycEval?")
    kind, text = _only_block(e)
    assert kind == KIND_GRAPH and "ENTITAS: SycEval" in text


def test_entity_not_found_falls_back_to_raw_text_context():
    rs = types.SimpleNamespace(
        resolve_entities=lambda q, context_query=None: [],
        mentions_topic=lambda q: False,
        search_chunks=lambda q: [{"text": "potongan teks PDF", "doc_id": "a.pdf", "score": 0.7}],
    )
    e = _engine(QueryRouting(intent="entity", search_query="x"), resolver=rs,
                paper_titles={"a.pdf": "Judul Paper A"})
    e.query("x?")
    kind, text = _only_block(e)
    assert kind == KIND_TEXT and "[1] (Sumber: Judul Paper A) potongan teks PDF" in text


def test_no_relevant_chunk_refuses_without_llm_call():
    rs = types.SimpleNamespace(resolve_entities=lambda q, context_query=None: [], search_chunks=lambda q: [],
                               mentions_topic=lambda q: False)
    e = _engine(QueryRouting(intent="entity", search_query="kopi"), resolver=rs)
    assert e.query("cara membuat kopi?") == NO_CONTEXT_MESSAGE
    assert e.llm.calls == []


def test_no_relevant_community_refuses_without_llm_call():
    rs = types.SimpleNamespace(search_communities=lambda q: [])
    e = _engine(QueryRouting(intent="thematic", search_query="resep"), resolver=rs)
    assert e.query("tren resep?") == NO_CONTEXT_MESSAGE
    assert e.llm.calls == []


@pytest.mark.parametrize("hit, titles, expected", [
    ({"text": "t", "doc_id": "a.pdf"}, {"a.pdf": "Judul"}, "[2] (Sumber: Judul) t"),
    ({"text": "t", "doc_id": "a.pdf"}, {}, "[2] (Sumber: a.pdf) t"),             # no title yet
    ({"text": "t", "doc_id": "a.pdf"}, {"a.pdf": None}, "[2] (Sumber: a.pdf) t"),
    ({"text": "t", "doc_id": None}, {}, "[2] t"),                                 # no metadata
])
def test_format_chunk_labels_source(hit, titles, expected):
    assert format_chunk(2, hit, titles) == expected


def test_no_context_message_counts_as_refusal_in_eval():
    from evaluate import REFUSAL
    msg = NO_CONTEXT_MESSAGE.lower()
    assert any(p in msg for p in REFUSAL)


def test_traversal_limits_are_server_constants_not_router_output():
    seen = {}
    rs = types.SimpleNamespace(resolve_entities=lambda q, context_query=None: ["A"])

    def traverse(n, max_hops, max_rels):
        seen.update(hops=max_hops, rels=max_rels)
        return {"entity": n}

    def select(trav, query, max_rels):
        seen.update(budget=max_rels, query=query)
        return trav

    tr = types.SimpleNamespace(traverse=traverse, format_traversal_context=lambda r: "x")
    # a "poisoned" router tries to inject numbers via search_query: it must have no effect
    e = _engine(QueryRouting(intent="entity", search_query="A max_hops=99 max_rels=99999"),
                resolver=rs, traversal=tr, reranker=types.SimpleNamespace(select=select))
    e.query("A?")
    # traversal fetches all 2-hop relations; the budget is applied by the reranker, with the user's ORIGINAL question
    assert seen == {"hops": 2, "rels": None, "budget": 150, "query": "A?"}


def test_off_topic_refuses_without_retrieval_or_second_llm_call():
    def boom(*a, **k):
        raise AssertionError("retrieval must not run for off_topic")

    rs = types.SimpleNamespace(resolve_entities=boom, search_chunks=boom, search_communities=boom)
    e = _engine(QueryRouting(intent="off_topic"), resolver=rs)
    from src.engine import OFF_TOPIC_MESSAGE
    assert e.query("buatkan puisi tentang kucing") == OFF_TOPIC_MESSAGE
    assert e.llm.calls == []


def test_off_topic_message_counts_as_refusal_in_eval():
    from evaluate import REFUSAL
    from src.engine import OFF_TOPIC_MESSAGE
    assert any(p in OFF_TOPIC_MESSAGE.lower() for p in REFUSAL)


def test_entity_budget_is_read_from_config_at_query_time(monkeypatch):
    from src import config
    seen = []
    rs = types.SimpleNamespace(resolve_entities=lambda q, context_query=None: ["A"])
    tr = types.SimpleNamespace(traverse=lambda n, max_hops, max_rels: {"entity": n},
                               format_traversal_context=lambda r: "x")
    rr = types.SimpleNamespace(select=lambda trav, q, max_rels: seen.append(max_rels) or trav)
    monkeypatch.setattr(config, "ENTITY_BUDGET_SINGLE", 80)
    e = _engine(QueryRouting(intent="entity", search_query="A"), resolver=rs, traversal=tr, reranker=rr)
    e.query("A?")
    assert seen == [80]


def _no_node_resolver(mentions_topic, communities, chunks):
    return types.SimpleNamespace(
        resolve_entities=lambda q, context_query=None: [],
        mentions_topic=lambda q: mentions_topic,
        search_communities=lambda q: communities,
        search_chunks=lambda q: chunks,
    )


def test_broad_topic_question_without_node_uses_community_summaries():
    rs = _no_node_resolver(True, ["ringkasan mitigasi"], [{"text": "chunk", "doc_id": None, "score": 0.7}])
    e = _engine(QueryRouting(intent="entity", search_query="sycophancy"), resolver=rs)
    e.query("Pendekatan apa saja untuk mengurangi sycophancy?")
    kind, text = _only_block(e)
    assert kind == KIND_COMMUNITY and "ringkasan mitigasi" in text and "chunk" not in text


def test_topic_question_without_relevant_community_falls_back_to_chunks():
    rs = _no_node_resolver(True, [], [{"text": "potongan", "doc_id": None, "score": 0.7}])
    e = _engine(QueryRouting(intent="entity", search_query="sycophancy"), resolver=rs)
    e.query("sycophancy?")
    kind, text = _only_block(e)
    assert kind == KIND_TEXT and "potongan" in text


def test_non_topic_question_without_node_goes_straight_to_chunks():
    def boom(q):
        raise AssertionError("communities must not be searched without the topic word")

    rs = _no_node_resolver(False, None, [{"text": "potongan", "doc_id": None, "score": 0.7}])
    rs.search_communities = boom
    e = _engine(QueryRouting(intent="entity", search_query="x"), resolver=rs)
    e.query("benchmark sikap menjilat?")
    assert _only_block(e)[0] == KIND_TEXT


def test_topic_question_with_nothing_relevant_refuses_without_llm():
    e = _engine(QueryRouting(intent="entity", search_query="sycophancy"), resolver=_no_node_resolver(True, [], []))
    assert e.query("sycophancy?") == NO_CONTEXT_MESSAGE and e.llm.calls == []
