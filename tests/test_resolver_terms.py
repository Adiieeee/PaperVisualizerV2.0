"""Matching node names inside a query (resolver layer 2 / multi-entity). No model/network."""
import pytest

from src.query.resolver import _contains_term


@pytest.mark.parametrize("term, text, expected", [
    ("rlhf", "penulis benchmark yang memakai dataset hh-rlhf?", False),   # adv-3hop: used to match
    ("gpt", "berapa harga api gpt-4o per token?", False),
    ("gpt-4o", "berapa harga api gpt-4o per token?", True),
    ("resnet-50", "apa itu resnet-50?", True),
    ("syceval", "dataset apa yang digunakan syceval?", True),
    ("gpt", "versi gpt apa saja yang diuji?", True),
    ("llama", "varian llama apa saja?", True),
])
def test_contains_term_treats_hyphen_as_part_of_word(term, text, expected):
    assert _contains_term(term, text) is expected
