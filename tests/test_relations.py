"""Skema relasi: apa yang ditawarkan ke LLM vs apa yang boleh tersimpan. Tanpa jaringan/kuota."""
import json
from typing import get_args

import pytest
from pydantic import ValidationError

from src import config
from src.graph.visualizer import GraphVisualizer
from src.llm import llm_service as svc

NEW = ("addresses_task", "introduces_dataset", "evaluated_on")


def _relation_enum(schema_model):
    return schema_model.model_json_schema()["$defs"]["LLMRelationship"]["properties"]["relation_type"]["enum"]


def test_llm_schema_offers_new_relations_but_not_related_to():
    enum = _relation_enum(config.LLMExtractedKnowledge)
    assert set(NEW) <= set(enum)
    assert "related_to" not in enum
    # tidak boleh bocor lewat deskripsi/docstring juga
    assert "related_to" not in json.dumps(config.LLMExtractedKnowledge.model_json_schema())


def test_stored_relationship_accepts_related_to_but_llm_one_does_not():
    rel = {"source_entity": "A", "target_entity": "B", "relation_type": "related_to"}
    assert config.Relationship(**rel).relation_type == "related_to"
    with pytest.raises(ValidationError):
        config.LLMRelationship(**rel)


def test_prompt_lists_exactly_the_llm_relations():
    p = svc.build_extraction_prompt("teks", "Judul")
    line = next(l for l in p.splitlines() if "RELATIONSHIP TYPES" in l)
    listed = [r.strip() for r in line.split(":", 1)[1].split(",")]
    assert listed == list(get_args(config.LLM_RELATION_TYPES))
    assert "related_to" not in p


def test_extraction_sends_llm_schema_and_returns_storage_model(monkeypatch):
    seen = {}

    def fake_call_llm(messages, alias="general", schema=None, **kw):
        seen["schema"] = schema
        out = {"entities": [{"name": "SycEval", "type": "Paper"}, {"name": "Math QA", "type": "Task"}],
               "relationships": [{"source_entity": "SycEval", "target_entity": "Math QA",
                                  "relation_type": "addresses_task"}]}
        if kw.get("validate") is False:
            return json.dumps(out), "gemini-test"
        return schema.model_validate(out), "gemini-test"

    monkeypatch.setattr(svc, "call_llm", fake_call_llm)
    monkeypatch.setattr(svc.time, "sleep", lambda s: None)
    k = svc.LLMService().extract_entities("teks", paper_title="SycEval").knowledge
    assert seen["schema"] is config.LLMExtractedKnowledge
    assert isinstance(k, config.ExtractedKnowledge)
    assert k.relationships[0].relation_type == "addresses_task"


@pytest.mark.parametrize("rel", list(get_args(config.RELATION_TYPES)) + ["variant_of"])
def test_every_relation_has_a_color_and_description(rel):
    assert rel in GraphVisualizer.EDGE_COLORS
    assert rel in GraphVisualizer.EDGE_DESCRIPTIONS
