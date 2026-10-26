"""Dedup when the daily quota runs out: the checkpoint is saved, the run stops, a resume continues. No network/quota."""
import json

import pytest

from src.config import DedupDecision, Entity, ExtractedKnowledge, Relationship
from src.extraction import deduplicator as D
from src.llm.llm_client import LLMQuotaError


class _FakeEmbed:
    """Every name has exactly the same vector -> every candidate passes the threshold and goes to the LLM."""

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
    llm = _FakeLLM(fail_on_call=2)  # LLM calls: B (ok), C (quota exhausted)
    dd = D.HybridDeduplicator(llm, _FakeEmbed())
    with pytest.raises(LLMQuotaError):
        dd.process_extracted_knowledge(_knowledge())
    ck = json.loads((paths / "checkpoint_dedup.json").read_text(encoding="utf-8"))
    # B was decided before the quota ran out -> saved even though item 1 is not finished
    assert set(ck["name_mapping"]) == {"A", "B"}
    assert ck["decision_models"] == {"B": "gemini-3.5-flash-lite"}


def test_resume_continues_from_interrupted_item_without_redoing_decisions(paths):
    with pytest.raises(LLMQuotaError):
        D.HybridDeduplicator(_FakeLLM(fail_on_call=2), _FakeEmbed()).process_extracted_knowledge(_knowledge())

    master, mapping = D._load_dedup_checkpoint()
    kl = _knowledge()
    resume_idx = D.dedup_resume_index(kl, mapping)  # used by main.py
    assert resume_idx == 1

    llm = _FakeLLM()
    dd = D.HybridDeduplicator(llm, _FakeEmbed())
    for t, ents in master.items():
        dd.master_list[t] = {n: {"embedding": d["embedding"]} for n, d in ents.items()}
    dd.name_mapping_global = mapping
    dd.decision_models = D._load_dedup_decision_models()
    dd.process_extracted_knowledge(kl, resume_from_idx=resume_idx)

    assert llm.calls == ["C", "D"]  # B is not decided again (does not burn quota again)
    ck = json.loads((paths / "checkpoint_dedup.json").read_text(encoding="utf-8"))
    assert set(ck["name_mapping"]) == {"A", "B", "C", "D"}
    assert set(ck["decision_models"]) == {"B", "C", "D"}


def test_old_dedup_checkpoint_without_models_still_loads(paths):
    (paths / "checkpoint_dedup.json").write_text(
        '{"master_list": {"Method": {"A": {"embedding": [1, 0]}}}, "name_mapping": {"A": "A"}}',
        encoding="utf-8")
    assert D._load_dedup_checkpoint()[1] == {"A": "A"}
    assert D._load_dedup_decision_models() == {}


# --- Resume: skipped items are still mapped, prefix (not count), normalization index ---

class _MergeLLM:
    """Merges every candidate into the top candidate (imitates the LLM's merge decision)."""

    def __init__(self):
        self.calls, self.last_model = [], "gemini-3.5-flash-lite"

    def resolve_dedup(self, cand, typ, candidates):
        self.calls.append(cand)
        return DedupDecision(is_duplicate=True, reason="sinonim", merged_name=candidates[0])


def _kl_with_rels():
    def k(a, b):
        return ExtractedKnowledge(
            entities=[Entity(name=a, type="Method"), Entity(name=b, type="Dataset")],
            relationships=[Relationship(source_entity=a, target_entity=b, relation_type="uses_dataset")])
    return [k("Stochastic Gradient Descent", "MedQuad"), k("SGD optimizer", "MedQuad")]


def _resume(kl, llm):
    master, mapping = D._load_dedup_checkpoint()
    dd = D.HybridDeduplicator(llm, _FakeEmbed())
    for t, ents in master.items():
        dd.master_list[t] = {n: {"embedding": d["embedding"]} for n, d in ents.items()}
    dd.name_mapping_global = mapping
    dd.rebuild_normalized_index()
    idx = D.dedup_resume_index(kl, mapping)
    dd.process_extracted_knowledge(kl, resume_from_idx=idx)
    return dd, idx


def test_resume_applies_saved_mapping_to_skipped_items(paths):
    # Full run: "SGD optimizer" is merged into "Stochastic Gradient Descent".
    D.HybridDeduplicator(_MergeLLM(), _FakeEmbed()).process_extracted_knowledge(_kl_with_rels())
    # Re-run (e.g. the build failed after dedup): knowledge is reloaded with the ORIGINAL names.
    kl = _kl_with_rels()
    llm = _MergeLLM()
    _, idx = _resume(kl, llm)
    assert idx == 2 and llm.calls == []  # everything already decided; no LLM
    assert [e.name for e in kl[1].entities] == ["Stochastic Gradient Descent", "MedQuad"]
    assert kl[1].relationships[0].source_entity == "Stochastic Gradient Descent"


def test_resume_index_is_prefix_not_count():
    def k(*names):
        return ExtractedKnowledge(entities=[Entity(name=n, type="Method") for n in names])
    kl = [k("A"), k("B", "C"), k("A")]  # item 1 interrupted (C not yet), item 2 happens to be complete
    assert D.dedup_resume_index(kl, {"A": "A", "B": "B"}) == 1  # a count of 2 would skip C


def test_resume_restores_normalization_index_so_no_llm_call(paths):
    def k(name):
        return ExtractedKnowledge(entities=[Entity(name=name, type="Method")])
    D.HybridDeduplicator(_MergeLLM(), _FakeEmbed()).process_extracted_knowledge([k("ResNet")])
    llm = _MergeLLM()
    dd, _ = _resume([k("ResNet"), k("resnet")], llm)
    assert llm.calls == []  # "resnet" matches deterministically (case), not via the LLM
    assert dd.name_mapping_global["resnet"] == "ResNet"
