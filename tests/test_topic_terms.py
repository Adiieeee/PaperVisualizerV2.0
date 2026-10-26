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
