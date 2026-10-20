import json
import os
import types

import networkx as nx
import pytest

from src import config
from src.security import integrity as I


class FakeCollection:
    def __init__(self, n, dim=384):
        self.n, self.dim = n, dim

    def count(self):
        return self.n

    def peek(self, k):
        return {"embeddings": [[0.0] * self.dim] if self.n else []}


def fake_chroma(chunks=189, summaries=30, dim=384):
    return types.SimpleNamespace(chunks_collection=FakeCollection(chunks, dim),
                                 community_collection=FakeCollection(summaries, dim))


class FakeEmbed:
    def __init__(self, dim=384):
        self.dim = dim

    def embed_text(self, t):
        return [0.0] * self.dim


@pytest.fixture
def art(tmp_path):
    G = nx.Graph()
    G.add_edge("a", "b", relation="uses_method")
    graph_path = str(tmp_path / "graph.graphml")
    nx.write_graphml(G, graph_path)
    return types.SimpleNamespace(G=G, graph_path=graph_path, manifest_path=str(tmp_path / "manifest.json"),
                                 chroma=fake_chroma())


def _write(art):
    m = I.build_manifest(art.graph_path, art.G, art.chroma)
    I.write_manifest(m, art.manifest_path)
    return m


# --- tulis atomik -----------------------------------------------------------------------------------

def test_atomic_write_json_roundtrip_and_no_tmp_left(tmp_path):
    p = str(tmp_path / "sub" / "c.json")
    I.atomic_write_json(p, {"k": "nilai unicode é"}, indent=2)
    assert json.load(open(p, encoding="utf-8")) == {"k": "nilai unicode é"}
    assert not os.path.exists(p + ".tmp")


def test_failed_write_keeps_previous_file_intact_and_cleans_tmp(tmp_path):
    p = str(tmp_path / "c.json")
    I.atomic_write_json(p, {"v": 1})

    def boom(tmp):
        with open(tmp, "w") as f:
            f.write('{"v": 2, "trunc')  # tulis setengah lalu crash
        raise RuntimeError("crash di tengah tulis")

    with pytest.raises(RuntimeError):
        I.atomic_write(p, boom)
    assert json.load(open(p)) == {"v": 1}      # file lama utuh: tidak pernah korup/terpotong
    assert not os.path.exists(p + ".tmp")


def test_atomic_write_for_graphml(tmp_path):
    G = nx.Graph()
    G.add_edge("a", "b")
    p = str(tmp_path / "g.graphml")
    I.atomic_write(p, lambda tmp: nx.write_graphml(G, tmp))
    assert nx.read_graphml(p).number_of_edges() == 1 and not os.path.exists(p + ".tmp")


# --- manifest ---------------------------------------------------------------------------------------

def test_manifest_records_model_revision_hash_and_counts(art):
    m = _write(art)
    assert m["embedding"]["model"] == config.EMBEDDING_MODEL_NAME
    assert m["embedding"]["revision"] == config.EMBEDDING_MODEL_REVISION
    assert m["embedding"]["dimension"] == 384
    assert m["graph"]["nodes"] == 2 and m["graph"]["edges"] == 1 and len(m["graph"]["sha256"]) == 64
    assert m["vectorstore"] == {"chunks": 189, "community_summaries": 30}


def test_consistent_artifacts_pass(art):
    _write(art)
    assert I.check_artifacts(art.graph_path, art.G, art.chroma, FakeEmbed(), art.manifest_path) is True


def test_missing_manifest_is_a_warning_not_an_error(art, capsys):
    assert I.check_artifacts(art.graph_path, art.G, art.chroma, FakeEmbed(), art.manifest_path) is False
    assert "manifest.json belum ada" in capsys.readouterr().out


def test_swapped_embedding_model_is_rejected_even_with_same_dimension(art, monkeypatch):
    """Inti D1: model lain berdimensi SAMA tidak crash — tanpa manifest jadi jawaban salah yang percaya diri."""
    _write(art)
    monkeypatch.setattr(config, "EMBEDDING_MODEL_NAME", "some/other-model-also-384d")
    with pytest.raises(I.ArtifactIntegrityError, match="Model embedding berubah"):
        I.check_artifacts(art.graph_path, art.G, art.chroma, FakeEmbed(384), art.manifest_path)


def test_changed_model_revision_is_rejected(art, monkeypatch):
    _write(art)
    monkeypatch.setattr(config, "EMBEDDING_MODEL_REVISION", "0" * 40)
    with pytest.raises(I.ArtifactIntegrityError, match="Revisi model"):
        I.check_artifacts(art.graph_path, art.G, art.chroma, FakeEmbed(), art.manifest_path)


def test_different_query_dimension_is_rejected(art):
    _write(art)
    with pytest.raises(I.ArtifactIntegrityError, match="Dimensi embedding"):
        I.check_artifacts(art.graph_path, art.G, art.chroma, FakeEmbed(768), art.manifest_path)


def test_modified_graph_file_is_rejected(art):
    _write(art)
    G2 = nx.Graph()
    G2.add_edge("a", "b", relation="uses_method")
    G2.add_edge("a", "SISIPAN", relation="uses_method")
    nx.write_graphml(G2, art.graph_path)  # file diubah setelah manifest ditulis
    with pytest.raises(I.ArtifactIntegrityError, match="berubah sejak manifest"):
        I.check_artifacts(art.graph_path, G2, art.chroma, FakeEmbed(), art.manifest_path)


def test_stale_chroma_content_is_rejected(art):
    _write(art)
    stale = fake_chroma(chunks=250)  # sisa chunk dari korpus lama (build tidak menghapus Chroma)
    with pytest.raises(I.ArtifactIntegrityError, match="clear"):
        I.check_artifacts(art.graph_path, art.G, stale, FakeEmbed(), art.manifest_path)


def test_schema_change_is_rejected(art, monkeypatch):
    _write(art)
    from typing import Literal
    monkeypatch.setattr(config, "ENTITY_TYPES", Literal["Paper", "Author", "TipeBaru"])
    with pytest.raises(I.ArtifactIntegrityError, match="tipe entitas"):
        I.check_artifacts(art.graph_path, art.G, art.chroma, FakeEmbed(), art.manifest_path)


def test_unreadable_manifest_is_a_clear_error(art):
    open(art.manifest_path, "w").write("{ bukan json")
    with pytest.raises(I.ArtifactIntegrityError, match="tidak bisa dibaca"):
        I.check_artifacts(art.graph_path, art.G, art.chroma, FakeEmbed(), art.manifest_path)


# --- checkpoint ekstraksi korup: berhenti dengan pesan jelas, jangan diam-diam mulai ulang ---------------------

def test_corrupt_extraction_checkpoint_raises_clear_error(tmp_path, monkeypatch):
    import main
    bad = tmp_path / "checkpoint_extraction.json"
    bad.write_text('{"done_chunk_ids": ["a"], "knowledge": [{"chunk_id": ', encoding="utf-8")  # terpotong
    monkeypatch.setattr(main, "CHECKPOINT_PATH", str(bad))
    with pytest.raises(I.ArtifactIntegrityError, match="corrupt"):
        main._load_checkpoint()


def test_extraction_checkpoint_roundtrip_is_atomic(tmp_path, monkeypatch):
    import main
    from src.config import Entity, ExtractedKnowledge
    p = tmp_path / "checkpoint_extraction.json"
    monkeypatch.setattr(main, "CHECKPOINT_PATH", str(p))
    k = ExtractedKnowledge(entities=[Entity(name="ResNet", type="Method")], relationships=[])
    main._save_checkpoint({"c0"}, [("c0", k)])
    done, items = main._load_checkpoint()
    assert done == {"c0"} and items[0][0] == "c0" and items[0][1].entities[0].name == "ResNet"
    assert not os.path.exists(str(p) + ".tmp")
