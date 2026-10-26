"""GraphAnalytics.get_usage_stats: peringkat counting menurut jumlah paper, lalu koneksi. Tanpa model."""
import networkx as nx

from src.graph.analytics import GraphAnalytics


def _graph():
    G = nx.Graph()
    for p in ("P1", "P2"):
        G.add_node(p, type="Paper")
    G.add_node("Hub", type="Dataset")      # banyak koneksi, satu paper
    G.add_node("Shared", type="Dataset")   # sedikit koneksi, dua paper
    G.add_node("Tie", type="Dataset")
    G.add_edge("Hub", "P1", relation="mentioned_in")
    for i in range(5):
        G.add_node(f"m{i}", type="Method")
        G.add_edge("Hub", f"m{i}", relation="evaluated_on")
    G.add_edge("Shared", "P1", relation="uses_dataset")
    G.add_edge("Shared", "P2", relation="uses_dataset")
    G.add_edge("Tie", "P2", relation="uses_dataset")
    G.add_edge("Tie", "m0", relation="variant_of")   # sintetis: tidak dihitung sebagai koneksi
    return G


def test_ranks_by_papers_then_connections():
    rows = GraphAnalytics(_graph()).get_usage_stats("Dataset")
    assert rows == [("Shared", 2, 2), ("Hub", 1, 5), ("Tie", 1, 1)]


def test_synthetic_relations_are_not_connections_but_mentioned_in_counts_a_paper():
    rows = dict((n, (p, d)) for n, p, d in GraphAnalytics(_graph()).get_usage_stats("Dataset"))
    assert rows["Hub"] == (1, 5)   # mentioned_in -> paper dihitung, tetapi bukan koneksi
    assert rows["Tie"] == (1, 1)   # variant_of tidak dihitung


def test_degree_ranking_is_unchanged_for_the_visualizer():
    assert GraphAnalytics(_graph()).get_counting_stats("Dataset")[0] == ("Hub", 5)
