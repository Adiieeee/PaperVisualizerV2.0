"""build_pipeline saat kuota harian habis SETELAH dedup: berhenti (bukan melewati diam-diam), checkpoint
tetap ada, dan re-run tidak mengulang dedup. Semua dependensi berat dipalsukan; tidak ada file di
data/processed yang disentuh. Tanpa jaringan/kuota."""
import os

import pytest

import main
from src.config import DedupDecision, Entity, ExtractedKnowledge, FamilyDecision, PaperMeta, Relationship
from src.extraction import deduplicator as D
from src.graph.builder import GraphBuilder
from src.llm.llm_client import DailyQuotaExceededError
from src.llm.llm_service import ExtractionResult

DOC = "paper.pdf"
CHUNKS = [{"chunk_id": f"{DOC}_chunk_{i}", "doc_id": DOC, "text": f"teks {i}"} for i in range(2)]


def _knowledge(i):
    method = ["Stochastic Gradient Descent", "SGD optimizer"][i]
    return ExtractedKnowledge(
        entities=[Entity(name=method, type="Method"), Entity(name="MedQuad", type="Dataset")],
        relationships=[Relationship(source_entity=method, target_entity="MedQuad", relation_type="uses_dataset")])


class _FakeLLM:
    def __init__(self, summary_quota=False):
        self.summary_quota, self.dedup_calls, self.last_model = summary_quota, [], "gemini-3.5-flash-lite"

    def extract_paper_metadata(self, text):
        self.metadata_calls = getattr(self, "metadata_calls", 0) + 1
        return PaperMeta(title="A Paper", models=[])

    def extract_entities(self, text, paper_title=None):
        return ExtractionResult(_knowledge(int(text.split()[-1])), "ok", 1)

    def resolve_dedup(self, cand, typ, candidates):
        self.dedup_calls.append(cand)
        return DedupDecision(is_duplicate=True, reason="sinonim", merged_name=candidates[0])

    def resolve_family_merge(self, *a):
        return FamilyDecision(same_family=False, canonical=a[0], reason="x")

    def summarize_community(self, text):
        if self.summary_quota:
            raise DailyQuotaExceededError("[DAILY QUOTA EXHAUSTED] Gemini + Groq")
        return "ringkasan"


class _FakeEmbed:
    def embed_text(self, text):
        return [1.0, 0.0]

    def cosine_similarity(self, a, b):
        return 1.0

    def embed_batch(self, texts):
        return [[1.0, 0.0] for _ in texts]


class _FakeCollection:
    def __init__(self, dim):
        self.dim = dim

    def count(self):
        return 1 if self.dim else 0

    def peek(self, n):
        return {"embeddings": [[0.0] * self.dim]}


class _FakeChroma:
    stored_dim = None  # dimensi vektor yang sudah tersimpan (None = store kosong)

    def __init__(self):
        self.chunks_collection = _FakeCollection(self.stored_dim)
        self.community_collection = _FakeCollection(None)

    def add_chunks(self, *a):
        pass

    def add_community_summaries(self, *a):
        pass


class _FakeAnalytics:
    def __init__(self, graph):
        pass

    def detect_communities(self):
        return [{"community_id": 0, "nodes": ["Stochastic Gradient Descent", "MedQuad"]}]


@pytest.fixture
def env(tmp_path, monkeypatch):
    state = {"llm": None, "saved_graphs": []}
    monkeypatch.setattr(main, "CHECKPOINT_PATH", str(tmp_path / "checkpoint_extraction.json"))
    monkeypatch.setattr(main, "EXTRACTION_MODELS_PATH", str(tmp_path / "extraction_models.json"))
    monkeypatch.setattr(D, "DEDUP_CHECKPOINT_PATH", str(tmp_path / "checkpoint_dedup.json"))
    monkeypatch.setattr(D, "SIMILARITY_DEBUG_PATH", str(tmp_path / "similarity_debug.csv"))
    monkeypatch.setattr(main.DocumentProcessor, "process_directory", lambda self, d: [dict(c) for c in CHUNKS])
    monkeypatch.setattr(main, "LLMService", lambda: state["llm"])
    monkeypatch.setattr(main, "EmbeddingClient", _FakeEmbed)
    monkeypatch.setattr(main, "load_paper_metadata", lambda: {})
    monkeypatch.setattr(main, "save_paper_metadata", lambda m: None)
    monkeypatch.setattr(main, "GraphAnalytics", _FakeAnalytics)
    monkeypatch.setattr(main, "ChromaStore", _FakeChroma)
    monkeypatch.setattr(main, "build_manifest", lambda *a: {})
    monkeypatch.setattr(main, "write_manifest", lambda m: None)
    monkeypatch.setattr(main, "_generate_visualization", lambda g: None)
    monkeypatch.setattr(GraphBuilder, "save_graph",
                        lambda self, path: state["saved_graphs"].append(set(self.graph.nodes)))
    state["tmp"] = tmp_path
    return state


def test_summary_quota_stops_build_keeps_checkpoints_and_rerun_skips_dedup(env):
    env["llm"] = _FakeLLM(summary_quota=True)
    main.build_pipeline()
    assert env["llm"].dedup_calls == ["SGD optimizer"]
    # Berhenti sebelum graph disimpan: graph/chroma/manifest lama tetap konsisten.
    assert env["saved_graphs"] == []
    assert (env["tmp"] / "checkpoint_extraction.json").exists()
    assert (env["tmp"] / "checkpoint_dedup.json").exists()

    env["llm"] = _FakeLLM()
    main.build_pipeline()
    assert env["llm"].dedup_calls == []  # dedup tidak diulang (tanpa kuota, keputusan sama)
    nodes = env["saved_graphs"][-1]
    assert "Stochastic Gradient Descent" in nodes and "SGD optimizer" not in nodes  # nama kanonik
    assert not (env["tmp"] / "checkpoint_extraction.json").exists()
    assert not (env["tmp"] / "checkpoint_dedup.json").exists()


def test_embedding_dimension_mismatch_stops_before_any_llm_call(env, monkeypatch):
    # Store dibangun model lain (384 dim); model palsu sekarang menghasilkan 2 dim.
    monkeypatch.setattr(_FakeChroma, "stored_dim", 384)
    env["llm"] = _FakeLLM()
    main.build_pipeline()
    assert getattr(env["llm"], "metadata_calls", 0) == 0 and env["llm"].dedup_calls == []
    assert not (env["tmp"] / "checkpoint_extraction.json").exists()


def test_matching_embedding_dimension_passes_preflight(env, monkeypatch):
    monkeypatch.setattr(_FakeChroma, "stored_dim", 2)  # sama dengan _FakeEmbed
    env["llm"] = _FakeLLM()
    main.build_pipeline()
    assert env["saved_graphs"]  # build berjalan sampai selesai


def test_family_merge_quota_stops_build(env, monkeypatch):
    def _quota(self, llm):
        raise DailyQuotaExceededError("[DAILY QUOTA EXHAUSTED] Gemini")
    monkeypatch.setattr(GraphBuilder, "merge_family_aliases", _quota)
    env["llm"] = _FakeLLM()
    main.build_pipeline()
    assert env["saved_graphs"] == []
    assert (env["tmp"] / "checkpoint_dedup.json").exists()


def test_builder_family_merge_reraises_quota_but_skips_other_errors(monkeypatch):
    b = GraphBuilder()
    b.graph.add_nodes_from(["ChatGPT", "GPT"])
    monkeypatch.setattr(GraphBuilder, "family_nodes", lambda self: {"ChatGPT": ["ChatGPT-4o"], "GPT": ["GPT-4"]})

    class _Llm:
        def __init__(self, exc):
            self.exc = exc

        def resolve_family_merge(self, *a):
            raise self.exc

    with pytest.raises(DailyQuotaExceededError):
        b.merge_family_aliases(_Llm(DailyQuotaExceededError("quota")))
    b.merge_family_aliases(_Llm(RuntimeError("parse/transient")))  # tetap dilewati, tidak raise
    assert not b.graph.has_edge("ChatGPT", "GPT")
