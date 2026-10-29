"""External background lookups (src/query/external.py): no network, HTTP and embeddings are faked."""
import json
import urllib.parse

import pytest

from src import config
from src.query import external as ext


def _wiki(*pages):
    return json.dumps({"query": {"pages": [
        {"index": i + 1, "title": p["title"], "extract": p.get("extract", "text"), "fullurl": f"https://en.wikipedia.org/wiki/{p['title']}",
         "lastrevid": 1, "description": p.get("description", ""), **({"pageprops": {"disambiguation": ""}} if p.get("disamb") else {})}
        for i, p in enumerate(pages)]}}).encode()


def _arxiv(*entries):
    body = "".join(f"<entry><id>http://arxiv.org/abs/{i}</id><title>{t}</title><summary>{s}</summary>"
                   f"<published>2021-09-08T00:00:00Z</published></entry>" for i, (t, s) in enumerate(entries))
    return f'<feed xmlns="http://www.w3.org/2005/Atom">{body}</feed>'.encode()


class FakeHTTP:
    def __init__(self, wiki=b'{"query": {}}', arxiv=b'<feed xmlns="http://www.w3.org/2005/Atom"/>', error=None):
        self.wiki, self.arxiv, self.error, self.urls = wiki, arxiv, error, []

    def __call__(self, url, timeout):
        self.urls.append(url)
        if self.error:
            raise self.error
        return self.wiki if url.startswith(ext.WIKI_API) else self.arxiv


class FakeEmbed:
    """Vector = [1, 0] for texts containing a GOOD marker, else [0, 1]; node text is [1, 0]."""
    def __init__(self, good=("GOOD",)):
        self.good = good

    def embed_batch(self, texts):
        return [[1.0, 0.0] if i == 0 or any(g in t for g in self.good) else [0.0, 1.0] for i, t in enumerate(texts)]


def _ctx(tmp_path, http, embed=None, clock=lambda: 1_000_000.0):
    return ext.ExternalContext(embed or FakeEmbed(), cache_path=str(tmp_path / "external_cache.json"), get=http, clock=clock)


def test_picks_matching_wikipedia_page_and_skips_disambiguation(tmp_path):
    http = FakeHTTP(wiki=_wiki({"title": "Gemini", "disamb": True, "extract": "GOOD constellation"},
                               {"title": "Gemini Dark Matter", "extract": "dark matter"},
                               {"title": "Gemini (language model)", "extract": "GOOD LLM by Google"}))
    e = _ctx(tmp_path, http).lookup("Gemini", "Gemini: An LLM.")
    assert e["title"] == "Gemini (language model)" and e["source"] == "wikipedia"


def test_title_gate_rejects_other_things_even_with_high_similarity(tmp_path):
    http = FakeHTTP(wiki=_wiki({"title": "Steering wheel", "extract": "GOOD"}))
    assert _ctx(tmp_path, http).lookup("contrastive activation steering", "x") is None


def test_low_similarity_candidate_is_rejected(tmp_path):
    http = FakeHTTP(wiki=_wiki({"title": "MMLU", "extract": "unrelated"}))
    assert _ctx(tmp_path, http).lookup("MMLU", "MMLU: A dataset.") is None


def test_title_tier():
    w = lambda t: {"source": "wikipedia", "title": t}
    a = lambda t: {"source": "arxiv", "title": t}
    assert ext.title_tier("Gemini", w("Gemini (language model)")) == 2
    assert ext.title_tier("Claude 2", w("Claude (AI)")) == 1          # version-like extra word
    assert ext.title_tier("Accuracy", w("Accuracy and precision")) == 1
    assert ext.title_tier("activation steering", w("Steering wheel")) == 0
    assert ext.title_tier("LLaMA 2", w("Llama.cpp")) == 0
    assert ext.title_tier("Claude Sonnet 3.7", w("Claude (AI)")) == 1
    # another concept that only shares a word, even a short one
    assert ext.title_tier("Social Sycophancy", w("Sycophancy (artificial intelligence)")) == 0
    assert ext.title_tier("Sycophancy Rate", w("Sycophancy (artificial intelligence)")) == 0
    assert ext.title_tier("TruthfulQA", a("TruthfulQA: Measuring How Models Mimic Human Falsehoods")) == 2
    assert ext.title_tier("DPO", a("What Matters in Data for DPO?")) == 1


def test_falls_back_to_arxiv_paper_named_after_the_entity(tmp_path):
    http = FakeHTTP(arxiv=_arxiv(("SymLoc: hallucination across TruthfulQA", "GOOD other"),
                                 ("TruthfulQA: Measuring How Models Mimic Human Falsehoods", "GOOD 817 questions")))
    e = _ctx(tmp_path, http).lookup("TruthfulQA", "TruthfulQA: A dataset.")
    assert e["source"] == "arxiv" and "817" in e["extract"] and e["url"].startswith("https://")


def test_arxiv_paper_merely_mentioning_the_entity_is_not_used(tmp_path):
    http = FakeHTTP(arxiv=_arxiv(("What Matters in Data for DPO?", "GOOD")))
    assert _ctx(tmp_path, http).lookup("DPO", "DPO: A method.") is None


def test_only_the_entity_name_is_sent(tmp_path):
    http = FakeHTTP()
    _ctx(tmp_path, http).lookup("Direct Preference Optimization (DPO)", "SECRET user question text")
    sent = " ".join(urllib.parse.unquote_plus(u) for u in http.urls)
    assert "Direct Preference Optimization" in sent and "(DPO)" not in sent and "SECRET" not in sent


@pytest.mark.parametrize("error", [TimeoutError("timed out"), OSError("HTTP Error 429"), ValueError("bad json")])
def test_errors_return_none_and_are_not_cached(tmp_path, error):
    c = _ctx(tmp_path, FakeHTTP(error=error))
    assert c.lookup("MMLU", "x") is None
    assert not (tmp_path / "external_cache.json").exists()


def test_cache_hit_needs_no_http_and_survives_restart(tmp_path):
    http = FakeHTTP(wiki=_wiki({"title": "MMLU", "extract": "GOOD benchmark"}))
    assert _ctx(tmp_path, http).lookup("MMLU", "x")["title"] == "MMLU"
    again = FakeHTTP(error=AssertionError("must not fetch"))
    assert _ctx(tmp_path, again).lookup("MMLU", "x")["title"] == "MMLU"
    assert again.urls == []


def test_not_found_is_cached_then_retried_after_its_ttl(tmp_path):
    now = [1_000_000.0]
    http = FakeHTTP()
    c = _ctx(tmp_path, http, clock=lambda: now[0])
    assert c.lookup("hh-rlhf", "x") is None
    n = len(http.urls)
    assert c.lookup("hh-rlhf", "x") is None and len(http.urls) == n          # cached "not found"
    now[0] += config.EXTERNAL_NOT_FOUND_TTL_DAYS * 86400 + 1
    c.lookup("hh-rlhf", "x")
    assert len(http.urls) > n                                                  # expired: fetched again


def test_found_entry_expires_after_ttl(tmp_path):
    now = [1_000_000.0]
    http = FakeHTTP(wiki=_wiki({"title": "MMLU", "extract": "GOOD"}))
    c = _ctx(tmp_path, http, clock=lambda: now[0])
    c.lookup("MMLU", "x")
    n = len(http.urls)
    now[0] += config.EXTERNAL_TTL_DAYS * 86400 + 1
    c.lookup("MMLU", "x")
    assert len(http.urls) > n


def test_extract_is_clipped():
    text = ("A sentence here. " * 100).strip()
    clipped = ext._clip(text, 100)
    assert len(clipped) <= 100 and clipped.endswith(".")


def test_corrupt_cache_is_an_empty_cache(tmp_path):
    p = tmp_path / "external_cache.json"
    p.write_text("{not json", encoding="utf-8")
    assert ext.load_cache(str(p)) == {}


def test_format_external_names_the_source():
    line = ext.format_external({"source": "wikipedia", "title": "MMLU", "extract": "A benchmark.", "url": "https://w/MMLU"})
    assert line == "[Wikipedia: MMLU] A benchmark. (https://w/MMLU)"
