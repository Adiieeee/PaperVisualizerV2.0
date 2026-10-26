"""Corpus topic terms and topic-only node names (resolver guard). No model/network."""
import pytest

from src.query.resolver import corpus_topic_terms, is_topic_only

TITLES = [
    "TOWARDS UNDERSTANDING SYCOPHANCY IN LANGUAGE MODELS",
    "SycEval: Evaluating LLM Sycophancy",
    "ELEPHANT: Measuring and understanding social sycophancy in LLMs",
    "What Counts as AI Sycophancy? A Taxonomy",
    "Beacon: Single-Turn Diagnosis of Latent Sycophancy",
    "Warning labels shift perceptions of sycophancy",
    "EchoBench: Benchmarking Sycophancy in Medical Large Vision-Language Models",
    "Be Friendly, Not Friends: How LLM Behaviour Shapes Trust",
]


def test_topic_terms_are_words_shared_by_most_titles():
    assert corpus_topic_terms(TITLES) == {"sycophancy"}


def test_stop_words_and_short_words_never_become_topic_terms():
    # "in" is in 4 of 8 titles and "llm" in 3: stop word / too short
    terms = corpus_topic_terms(TITLES, min_share=0.3)
    assert "in" not in terms and "llm" not in terms and "sycophancy" in terms


def test_no_topic_terms_below_the_paper_minimum_or_without_shared_words():
    assert corpus_topic_terms(TITLES[:2]) == frozenset()
    assert corpus_topic_terms(["Highway Networks", "Deep Residual Learning", "Going deeper with convolutions"]) \
        == frozenset()
    assert corpus_topic_terms([]) == frozenset()


@pytest.mark.parametrize("name, expected", [
    ("Sycophancy", True),
    ("AI Sycophancy", True),               # "ai" is too short to be a content word
    ("Social Sycophancy", False),          # a real concept: content word "social"
    ("Sycophancy Rate", False),
    ("Validation sycophancy", False),
    ("ELEPHANT", False),                   # no topic word at all
    ("sycophancy alter users social judgments", False),
])
def test_topic_only_node_names(name, expected):
    assert is_topic_only(name, frozenset({"sycophancy"})) is expected


def test_without_topic_terms_nothing_is_topic_only():
    assert is_topic_only("Sycophancy", frozenset()) is False


# ── resolver guard (mini graph, fake embeddings) ─────────────────────────────────────────────────────

import types

import networkx as nx

from src.query.resolver import QueryResolver


def _resolver(topic_terms):
    G = nx.Graph()
    for n, t in [("Sycophancy", "Finding"), ("AI Sycophancy", "Method"), ("ELEPHANT", "Method"),
                 ("Social Sycophancy", "Task"), ("Sycophancy Rate", "Metric"), ("P", "Paper")]:
        G.add_node(n, type=t)
    r = object.__new__(QueryResolver)
    r.graph, r.model_to_paper = G, {}
    r.topic_terms = frozenset(topic_terms)
    r._topic_only = frozenset(n for n in G if is_topic_only(n, r.topic_terms))
    r._node_names = r._node_embs = None
    # similarity 0 to every node: layer 4 never matches, so only the name-based layers are under test
    r.embed = types.SimpleNamespace(embed_text=lambda q: [1.0], embed_batch=lambda ns: [[1.0] for _ in ns],
                                    cosine_similarity=lambda a, b: 0.0)
    return r


@pytest.mark.parametrize("q", ["sycophancy", "Pendekatan apa saja untuk mengurangi sycophancy?",
                               "Apa itu AI sycophancy?"])
def test_topic_only_nodes_are_never_resolved(q):
    r = _resolver({"sycophancy"})
    assert r.resolve_entities(q, context_query=q) == []


@pytest.mark.parametrize("q, expected", [
    ("Bagaimana ELEPHANT mengukur sycophancy?", ["ELEPHANT"]),        # the real entity, alone and first
    ("Bagaimana social sycophancy diukur?", ["Social Sycophancy"]),
    ("Apa itu sycophancy rate?", ["Sycophancy Rate"]),
])
def test_real_entities_next_to_the_topic_word_still_resolve(q, expected):
    r = _resolver({"sycophancy"})
    assert r.resolve_entities(q, context_query=q) == expected


def test_without_topic_terms_behaviour_is_unchanged():
    r = _resolver(set())
    q = "Bagaimana ELEPHANT mengukur sycophancy?"
    assert r.resolve_entities(q, context_query=q) == ["Sycophancy", "ELEPHANT"]   # old behaviour
    assert r.resolve_entities("sycophancy", context_query="sycophancy") == ["Sycophancy"]
    assert r.embedding_probe("apa itu sycophancy?") == "sycophancy"
    assert r.mentions_topic("apa itu sycophancy?") is False


def test_embedding_probe_drops_topic_terms():
    r = _resolver({"sycophancy"})
    assert r.embedding_probe("mengurangi sycophancy pada chatbot") == "mengurangi pada chatbot"
    assert r.embedding_probe("sycophancy?") is None          # nothing but the topic: no probe
    assert "Sycophancy" not in r.embedding_candidates() and "ELEPHANT" in r.embedding_candidates()


def test_mentions_topic_matches_whole_words():
    r = _resolver({"sycophancy"})
    assert r.mentions_topic("Apa dampak Sycophancy?") is True
    assert r.mentions_topic("benchmark sikap menjilat") is False
