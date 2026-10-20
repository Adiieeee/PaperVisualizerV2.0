"""Dedup saat kuota harian habis: checkpoint tersimpan, run berhenti, resume melanjutkan. Tanpa jaringan/kuota."""
import json

import pytest

from src.config import DedupDecision, Entity, ExtractedKnowledge
from src.extraction import deduplicator as D
from src.llm.llm_client import LLMQuotaError


class _FakeEmbed:
    """Semua nama sama persis vektornya -> setiap kandidat lolos threshold dan masuk ke LLM."""

    def embed_text(self, text):
        return [1.0, 0.0]

    def cosine_similarity(self, a, b):
        return 1.0


class _FakeLLM:
    def __init__(self, fail_on_call=None):
        self.fail_on_call, self.calls, self.last_model = fail_on_call, [], None

    def resolve_dedup(self, cand, typ, candidates):
        self.calls.append(cand)
        if len(self.calls) == self.fail_on_call:
            raise LLMQuotaError("kuota harian habis")
        self.last_model = "gemini-3.5-flash-lite"
        return DedupDecision(is_duplicate=False, reason="beda", merged_name=cand)


def _knowledge():
    def k(*names):
        return ExtractedKnowledge(entities=[Entity(name=n, type="Method") for n in names])
    return [k("A"), k("B", "C"), k("D")]


@pytest.fixture
def paths(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "DEDUP_CHECKPOINT_PATH", str(tmp_path / "checkpoint_dedup.json"))
    monkeypatch.setattr(D, "SIMILARITY_DEBUG_PATH", str(tmp_path / "similarity_debug.csv"))
    return tmp_path


def test_quota_mid_item_saves_checkpoint_and_reraises(paths):
    llm = _FakeLLM(fail_on_call=2)  # panggilan LLM: B (ok), C (kuota habis)
    dd = D.HybridDeduplicator(llm, _FakeEmbed())
    with pytest.raises(LLMQuotaError):
        dd.process_extracted_knowledge(_knowledge())
    ck = json.loads((paths / "checkpoint_dedup.json").read_text(encoding="utf-8"))
    # B sudah diputuskan sebelum kuota habis -> ikut tersimpan walau item 1 belum selesai
    assert set(ck["name_mapping"]) == {"A", "B"}
    assert ck["decision_models"] == {"B": "gemini-3.5-flash-lite"}


def test_resume_continues_from_interrupted_item_without_redoing_decisions(paths):
    with pytest.raises(LLMQuotaError):
        D.HybridDeduplicator(_FakeLLM(fail_on_call=2), _FakeEmbed()).process_extracted_knowledge(_knowledge())

    master, mapping = D._load_dedup_checkpoint()
    kl = _knowledge()
    resume_idx = sum(1 for k in kl if all(e.name in mapping for e in k.entities))  # = main.py
    assert resume_idx == 1

    llm = _FakeLLM()
    dd = D.HybridDeduplicator(llm, _FakeEmbed())
    for t, ents in master.items():
        dd.master_list[t] = {n: {"embedding": d["embedding"]} for n, d in ents.items()}
    dd.name_mapping_global = mapping
    dd.decision_models = D._load_dedup_decision_models()
    dd.process_extracted_knowledge(kl, resume_from_idx=resume_idx)

    assert llm.calls == ["C", "D"]  # B tidak diputuskan ulang (tidak membakar kuota lagi)
    ck = json.loads((paths / "checkpoint_dedup.json").read_text(encoding="utf-8"))
    assert set(ck["name_mapping"]) == {"A", "B", "C", "D"}
    assert set(ck["decision_models"]) == {"B", "C", "D"}


def test_old_dedup_checkpoint_without_models_still_loads(paths):
    (paths / "checkpoint_dedup.json").write_text(
        '{"master_list": {"Method": {"A": {"embedding": [1, 0]}}}, "name_mapping": {"A": "A"}}',
        encoding="utf-8")
    assert D._load_dedup_checkpoint()[1] == {"A": "A"}
    assert D._load_dedup_decision_models() == {}

