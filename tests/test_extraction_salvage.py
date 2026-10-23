"""Ekstraksi: ulang sekali -> selamatkan sebagian (item valid terbanyak) -> gagal. Tanpa jaringan/kuota."""
import json

import pytest

from src.llm import llm_service as svc

P = "SycEval: Evaluating LLM Sycophancy"
LIM = "The reliance on synthetic rebuttals may not fully capture real-world interaction diversity."


def _out(entities, relationships=()):
    return json.dumps({"entities": list(entities), "relationships": list(relationships)})


VALID = _out([{"name": P, "type": "Paper"}, {"name": LIM, "type": "Limitation"}],
             [{"source_entity": P, "target_entity": LIM, "relation_type": "has_limitation"}])
# satu relasi di luar daftar -> seluruh output gagal validasi ketat
BAD_REL = _out([{"name": P, "type": "Paper"}, {"name": "MedQuad", "type": "Dataset"}, {"name": LIM, "type": "Limitation"}],
               [{"source_entity": P, "target_entity": "MedQuad", "relation_type": "evaluates", "description": "on MedQuad"},
                {"source_entity": P, "target_entity": LIM, "relation_type": "has_limitation"}])
FEWER_BAD = _out([{"name": P, "type": "Paper"}, {"name": "X", "type": "Gadget"}])  # 1 entitas valid, 1 tipe salah


class _Seq:
    """call_llm palsu: tiap panggilan mengembalikan teks berikutnya dari daftar."""

    def __init__(self, outputs):
        self.outputs, self.calls = list(outputs), []

    def __call__(self, messages, alias="general", schema=None, **kw):
        self.calls.append({"alias": alias, "schema": schema, **kw})
        assert kw.get("validate") is False and schema is not None  # skema tetap dikirim, teks mentah kembali
        return self.outputs[len(self.calls) - 1], "gemini-test"


def _run(monkeypatch, outputs):
    seq = _Seq(outputs)
    monkeypatch.setattr(svc, "call_llm", seq)
    monkeypatch.setattr(svc.time, "sleep", lambda s: None)
    return svc.LLMService().extract_entities("teks", paper_title=P), seq


def test_valid_first_try_is_ok_with_one_call(monkeypatch):
    r, seq = _run(monkeypatch, [VALID])
    assert (r.status, r.attempts, len(seq.calls)) == ("ok", 1, 1)
    assert [e.name for e in r.knowledge.entities] == [P, LIM]


def test_invalid_then_valid_is_retried(monkeypatch):
    r, seq = _run(monkeypatch, [BAD_REL, VALID])
    assert (r.status, r.attempts, len(seq.calls)) == ("retried", 2, 2)
    assert r.knowledge.relationships[0].relation_type == "has_limitation"


def test_invalid_twice_salvages_the_output_with_most_valid_items(monkeypatch):
    r, seq = _run(monkeypatch, [FEWER_BAD, BAD_REL])  # output ke-2 punya lebih banyak item valid
    assert (r.status, len(seq.calls)) == ("salvaged", 2)
    assert [e.name for e in r.knowledge.entities] == [P, "MedQuad", LIM]
    rels = {rel.relation_type: rel for rel in r.knowledge.relationships}
    assert rels["related_to"].description == "[evaluates] on MedQuad"  # label asli disimpan
    assert "has_limitation" in rels
    assert r.dropped == 0  # relasi tak dikenal dipetakan, bukan dibuang


def test_unknown_entity_type_is_dropped_and_counted():
    k, dropped = svc.salvage_knowledge(FEWER_BAD)
    assert [e.name for e in k.entities] == [P] and dropped == 1


@pytest.mark.parametrize("raw", ["", "bukan json", '{"entities": [', "[1, 2]", "null"])
def test_unreadable_output_cannot_be_salvaged(raw):
    assert svc.salvage_knowledge(raw) is None


def test_unreadable_twice_is_failed_after_exactly_two_calls(monkeypatch):
    r, seq = _run(monkeypatch, ['{"entities": [', ""])
    assert (r.status, r.attempts, len(seq.calls)) == ("failed", 2, 2)
    assert r.knowledge.entities == [] and r.knowledge.relationships == []


def test_valid_empty_output_is_ok_not_failed(monkeypatch):
    r, _ = _run(monkeypatch, [_out([])])  # chunk tanpa entitas (mis. hanya caption) itu sah
    assert r.status == "ok" and r.knowledge.entities == []


def test_relationship_with_missing_endpoint_is_dropped():
    raw = _out([{"name": P, "type": "Paper"}], [{"target_entity": P, "relation_type": "evaluates"}])
    k, dropped = svc.salvage_knowledge(raw)
    assert k.relationships == [] and dropped == 1
