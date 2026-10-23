"""Community detection over the chunk co-occurrence graph (graph-only, no LLM/torch)."""
import networkx as nx

from src.graph.analytics import GraphAnalytics
from src.vectorstore.chromadb_store import ChromaStore


def _graph():
    G = nx.Graph()
    G.add_node("Paper A", type="Paper", source_chunk_id="")
    G.add_node("Alice", type="Author", source_chunk_id="a_chunk_0")
    G.add_node("Uni X", type="Institution", source_chunk_id="a_chunk_0")
    # Three entities that only touch the Paper, but share chunk a_chunk_1.
    for n, t in [("ELEPHANT", "Method"), ("AITA-YTA", "Dataset"), ("OEQ", "Dataset")]:
        G.add_node(n, type=t, description="d " * 100, source_chunk_id="")
        G.add_edge("Paper A", n, relation="uses_dataset", source_chunk_id="a_chunk_1")
    G.add_edge("ELEPHANT", "AITA-YTA", relation="uses_dataset", source_chunk_id="a_chunk_1")
    G.add_edge("Paper A", "Alice", relation="written_by", source_chunk_id="a_chunk_0")
    G.add_edge("Alice", "Uni X", relation="affiliated_with", source_chunk_id="a_chunk_0")
    # A pair in another chunk: below the minimum size.
    G.add_node("MedQuad", type="Dataset", source_chunk_id="a_chunk_5")
    G.add_node("SycEval", type="Method", source_chunk_id="a_chunk_5")
    G.add_edge("Paper A", "MedQuad", relation="mentioned_in", source_chunk_id="a_chunk_5")
    return G


def test_entities_sharing_a_chunk_form_one_community():
    comms = GraphAnalytics(_graph()).detect_communities()
    assert comms == [{"community_id": 0, "nodes": ["AITA-YTA", "ELEPHANT", "OEQ"]}]


def test_excluded_types_never_members():
    comms = GraphAnalytics(_graph()).detect_communities(min_size=1)
    members = {n for c in comms for n in c["nodes"]}
    assert not members & {"Paper A", "Alice", "Uni X"}
    assert {"MedQuad", "SycEval"} <= members  # pair found, only dropped by min_size


def test_ids_sequential_and_deterministic():
    a = GraphAnalytics(_graph())
    first, second = a.detect_communities(min_size=1), a.detect_communities(min_size=1)
    assert first == second
    assert [c["community_id"] for c in first] == list(range(len(first)))
    assert [len(c["nodes"]) for c in first] == sorted((len(c["nodes"]) for c in first), reverse=True)


def test_community_context_has_types_descriptions_relations():
    G = _graph()
    G.add_edge("OEQ", "AITA-YTA", relation="mentioned_in", source_chunk_id="a_chunk_1")
    text = GraphAnalytics(G).community_context(["AITA-YTA", "ELEPHANT", "OEQ"])
    assert "- ELEPHANT (Method): d d" in text
    assert "..." in text and "d " * 100 not in text  # description truncated
    assert "ELEPHANT -[uses_dataset]- AITA-YTA" in text or "AITA-YTA -[uses_dataset]- ELEPHANT" in text
    assert "mentioned_in" not in text


def test_add_community_summaries_replaces_old_ones(tmp_path):
    store = ChromaStore(persist_directory=str(tmp_path / "chroma"))
    emb = [[1.0, 0.0]]
    store.add_community_summaries([{"community_id": i, "summary": f"s{i}"} for i in range(3)], emb * 3)
    store.add_community_summaries([{"community_id": 0, "summary": "new"}], emb)
    got = store.community_collection.get(include=["documents"])
    assert got["ids"] == ["0"] and got["documents"] == ["new"]
    store.add_community_summaries([], [])
    assert store.community_collection.count() == 0
