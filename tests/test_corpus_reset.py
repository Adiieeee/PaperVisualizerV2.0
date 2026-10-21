"""corpus.py reset-index: hapus indeks hasil build, pertahankan PDF + paper_metadata.json. Tanpa jaringan."""
import types

import pytest

import corpus


@pytest.fixture
def tree(tmp_path, monkeypatch):
    raw, processed = tmp_path / "raw", tmp_path / "processed"
    raw.mkdir(); processed.mkdir()
    (raw / "a.pdf").write_bytes(b"%PDF")
    for name in ("paper_metadata.json", "graph.graphml", "manifest.json", "graph_visualization.html",
                 "similarity_debug.csv", "checkpoint_extraction.json", "checkpoint_dedup.json",
                 "extraction_models.json", ".gitkeep"):
        (processed / name).write_text("x", encoding="utf-8")
    (processed / "chroma_db").mkdir()
    (processed / "chroma_db" / "chroma.sqlite3").write_text("x", encoding="utf-8")
    monkeypatch.setattr(corpus, "RAW", raw)
    monkeypatch.setattr(corpus, "PROCESSED", processed)
    return raw, processed


def test_reset_index_keeps_pdfs_and_paper_metadata(tree):
    raw, processed = tree
    corpus.cmd_reset_index(types.SimpleNamespace(yes=True))
    assert (raw / "a.pdf").exists()
    assert sorted(p.name for p in processed.iterdir()) == [".gitkeep", "paper_metadata.json"]


def test_reset_index_asks_and_can_be_cancelled(tree, monkeypatch):
    _, processed = tree
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    corpus.cmd_reset_index(types.SimpleNamespace(yes=False))
    assert (processed / "graph.graphml").exists() and (processed / "chroma_db").exists()
