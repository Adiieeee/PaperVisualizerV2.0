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


@pytest.mark.parametrize("case, text, retrieval_ok, e2e_ok", [
    ({"expected_contains": ["a", "b"]}, "a b", True, True),                        # kasus lama: semua harus ada
    ({"expected_contains": ["a", "b"]}, "a", False, False),
    ({"expected_any": ["x", "y"]}, "ada y", True, True),                           # salah satu cukup
    ({"expected_any": ["x", "y"]}, "kosong", False, False),
    ({"expected_contains": ["a"], "expected_any": ["x"]}, "a", False, False),      # keduanya berlaku
    ({"expected_contains": ["mitigat"], "e2e_any": ["DPO", "mitigasi"]}, "mitigasi", False, True),  # override e2e
    ({"expected_contains": ["mitigat"], "e2e_any": ["DPO"]}, "mitigation", True, False),
    ({"expected_contains": ["x"], "e2e_contains": []}, "apa saja", False, True),   # override kosong = lolos
    ({}, "apa saja", True, True),
])
def test_facts_match_all_any_and_e2e_override(case, text, retrieval_ok, e2e_ok):
    assert evaluate.facts_match(case, text) is retrieval_ok
    assert evaluate.facts_match(case, text, e2e=True) is e2e_ok


def test_e2e_grading_uses_the_e2e_keys(monkeypatch, capsys):
    cases = [{"id": "m", "q": "qm", "expected_contains": ["mitigat"], "e2e_any": ["DPO", "steering"]}]
    _fake_engine(monkeypatch, {"qm": "Strategi mitigasi: DPO-Validation"})
    evaluate.run_e2e(cases, "g")
    assert "[PASS" in capsys.readouterr().out


@pytest.mark.parametrize("top, text, ok", [
    (None, "B: 1 papers, 9 ; A: 2 papers, 3", True),
    (["A"], "A: 2 papers, 3 ; B: 1 papers, 9", True),
    (["A"], "B: 1 papers, 9 ; A: 2 papers, 3", False),   # ada di top-10 tetapi bukan peringkat pertama
    (["X", "B"], "B: 1 papers, 9 ; A: 2 papers, 3", True),  # seri: salah satu cukup
])
def test_expected_top_checks_the_first_ranked_item(top, text, ok):
    case = {"expected_top": top} if top else {}
    assert evaluate.top_match(case, text) is ok
    assert evaluate.grade(case, text, None)[0] is ok


def test_repeat_reports_per_case_rate_and_unstable_cases(monkeypatch, capsys):
    answers = iter(["pakai MedQuad", "salah", "pakai MedQuad"])

    class _Engine:
        def __init__(self, graph_path):
            self.llm = types.SimpleNamespace(last_model=None)

        def query(self, q):
            self.llm.last_model = "gemini-flash-lite-latest"
            return next(answers)

    monkeypatch.setattr(src.engine, "GraphRAGEngine", _Engine)
    evaluate.run_e2e([{"id": "a", "q": "qa", "expected_contains": ["MedQuad"]}], "g", repeat=3)
    out = capsys.readouterr().out
    assert "a#1" in out and "a#3" in out
    assert "E2E OVERALL: 2/3" in out and "a 2/3" in out and "Unstable (passed some runs, not all): 1: a" in out


def test_quota_stops_all_remaining_repeats(monkeypatch, capsys):
    _fake_engine(monkeypatch, {"qa": DailyQuotaExceededError("habis"), "qb": "x", "qc": "x"})
    evaluate.run_e2e(CASES, "g", repeat=2)
    out = capsys.readouterr().out
    assert out.count("[QUOTA") == 1 and "Not run: 6 case(s)" in out


def test_only_keeps_eval_order_and_rejects_unknown_ids():
    cases = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    assert [c["id"] for c in evaluate.select_cases(cases, "c, a")] == ["a", "c"]
    assert evaluate.select_cases(cases, None) == cases
    with pytest.raises(SystemExit, match="zz"):
        evaluate.select_cases(cases, "a,zz")


def test_budget_override_scales_multi_entity_budget(monkeypatch):
    from src import config
    monkeypatch.setattr(config, "ENTITY_BUDGET_SINGLE", 150)
    monkeypatch.setattr(config, "ENTITY_BUDGET_MULTI", 125)
    evaluate.set_budget(80)
    assert (config.ENTITY_BUDGET_SINGLE, config.ENTITY_BUDGET_MULTI) == (80, 67)
