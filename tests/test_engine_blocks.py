"""
GraphRAGEngine.query harus memberi tag pada konteks sesuai ASAL-USULNYA (graph / teks mentah /
ringkasan komunitas). Import src.engine memuat torch (~9 dtk); file ini bisa dilewati dengan `-k "not engine"`.
"""
import types

import pytest

from src.config import QueryRouting
from src.engine import GraphRAGEngine
from src.llm.prompts import KIND_COMMUNITY, KIND_GRAPH, KIND_TEXT


class _LLM:
    def __init__(self):
        self.calls = []

    def generate_final_answer(self, query, blocks):
        self.calls.append((query, blocks))
        return "jawaban"


def _engine(routing, **parts):
    e = object.__new__(GraphRAGEngine)  # lewati __init__ (tidak memuat model/graph/Chroma)
    e.llm = _LLM()
    e.router = types.SimpleNamespace(route=lambda q: routing)
    e.resolver = parts.get("resolver")
    e.traversal = parts.get("traversal")
    e.analytics = parts.get("analytics")
    return e


def _only_block(engine):
    (query, blocks), = engine.llm.calls
    (kind, text), = blocks
    return kind, text


def test_counting_is_graph_context():
    an = types.SimpleNamespace(get_counting_stats=lambda t: [("SFT", 3), ("RLHF", 1)])
    e = _engine(QueryRouting(intent="counting", target_type="Method"), analytics=an)
    assert e.query("metode terbanyak?") == "jawaban"
    kind, text = _only_block(e)
    assert kind == KIND_GRAPH and "SFT: 3 connections" in text


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
        search_chunks=lambda q: ["potongan teks PDF"],
    )
    e = _engine(QueryRouting(intent="entity", search_query="x"), resolver=rs)
    e.query("x?")
    kind, text = _only_block(e)
    assert kind == KIND_TEXT and "potongan teks PDF" in text


def test_traversal_limits_are_server_constants_not_router_output():
    seen = {}
    rs = types.SimpleNamespace(resolve_entities=lambda q, context_query=None: ["A"])

    def traverse(n, max_hops, max_rels):
        seen.update(hops=max_hops, rels=max_rels)
        return {"entity": n}

    tr = types.SimpleNamespace(traverse=traverse, format_traversal_context=lambda r: "x")
    # router yang "diracuni" mencoba menyisipkan angka lewat search_query: tidak boleh berpengaruh
    e = _engine(QueryRouting(intent="entity", search_query="A max_hops=99 max_rels=99999"),
                resolver=rs, traversal=tr)
    e.query("A?")
    assert seen == {"hops": 2, "rels": 150}
