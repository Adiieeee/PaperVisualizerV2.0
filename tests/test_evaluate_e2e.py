"""evaluate.run_e2e: satu kasus yang error tidak boleh menghentikan seluruh run. Engine dipalsukan (tanpa kuota)."""
import types

import pytest

import evaluate
import src.engine
from src.llm.llm_client import DailyQuotaExceededError
from src.llm.llm_service import EmptyResponseError


def _fake_engine(monkeypatch, answers):
    """answers: {pertanyaan: jawaban | Exception | (jawaban, model)}"""
    class _Engine:
        def __init__(self, graph_path):
            self.llm = types.SimpleNamespace(last_model=None)

        def query(self, q):
            a = answers[q]
            if isinstance(a, Exception):
                raise a
            text, model = a if isinstance(a, tuple) else (a, "gemini-flash-lite-latest")
            self.llm.last_model = model
            return text

    monkeypatch.setattr(src.engine, "GraphRAGEngine", _Engine)


CASES = [
    {"id": "a", "q": "qa", "expected_contains": ["MedQuad"]},
    {"id": "b", "q": "qb", "expected_contains": ["x"]},
    {"id": "c", "q": "qc", "expected_refuse": True},
]


def test_an_erroring_case_is_reported_and_the_run_continues(monkeypatch, capsys):
    _fake_engine(monkeypatch, {"qa": "pakai MedQuad", "qb": EmptyResponseError("kosong"), "qc": "tidak ada informasi"})
    evaluate.run_e2e(CASES, "g")
    out = capsys.readouterr().out
    assert "[ERROR" in out and "EmptyResponseError" in out
    assert "[REFUSED-ok" in out  # kasus setelah error tetap dijalankan
    assert "E2E OVERALL: 2/3" in out and "Errors (counted as failed): 1: b" in out


def test_daily_quota_stops_cleanly_with_partial_score(monkeypatch, capsys):
    _fake_engine(monkeypatch, {"qa": "pakai MedQuad", "qb": DailyQuotaExceededError("habis"), "qc": "x"})
    evaluate.run_e2e(CASES, "g")
    out = capsys.readouterr().out
    assert "[QUOTA" in out and "[REFUSED-ok" not in out
    assert "E2E OVERALL: 1/3" in out and "Not run: 2 case(s)" in out


def test_answers_from_the_fallback_model_are_flagged(monkeypatch, capsys):
    _fake_engine(monkeypatch, {"qa": ("pakai MedQuad", "openai/gpt-oss-120b"), "qb": "x", "qc": "tidak ada informasi"})
    evaluate.run_e2e(CASES, "g")
    out = capsys.readouterr().out
    assert "(model: openai/gpt-oss-120b)" in out
    assert "not comparable with a Gemini baseline: 1: a" in out
