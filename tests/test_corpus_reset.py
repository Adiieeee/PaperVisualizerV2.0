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
                 "extraction_models.json", "extraction_cache.json", "fact_embeddings.npz", ".gitkeep"):
        (processed / name).write_text("x", encoding="utf-8")
    (processed / "chroma_db").mkdir()
    (processed / "chroma_db" / "chroma.sqlite3").write_text("x", encoding="utf-8")
    monkeypatch.setattr(corpus, "RAW", raw)
    monkeypatch.setattr(corpus, "PROCESSED", processed)
    return raw, processed


def test_reset_index_keeps_pdfs_paper_metadata_and_extraction_archive(tree):
    raw, processed = tree
    corpus.cmd_reset_index(types.SimpleNamespace(yes=True))
    assert (raw / "a.pdf").exists()
    assert sorted(p.name for p in processed.iterdir()) == [
        ".gitkeep", "extraction_cache.json", "paper_metadata.json"]


def test_reset_index_asks_and_can_be_cancelled(tree, monkeypatch):
    _, processed = tree
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    corpus.cmd_reset_index(types.SimpleNamespace(yes=False))
    assert (processed / "graph.graphml").exists() and (processed / "chroma_db").exists()


def test_clear_deletes_the_extraction_archive(tree):
    _, processed = tree
    corpus.cmd_clear(types.SimpleNamespace(yes=True))
    assert not (processed / "extraction_cache.json").exists()
    assert not (processed / "fact_embeddings.npz").exists()


def test_backup_includes_the_extraction_archive(tree, tmp_path, monkeypatch):
    monkeypatch.setattr(corpus, "BACKUPS", tmp_path / "backups")
    monkeypatch.setattr(corpus, "EVAL_CASES", tmp_path / "no_eval_cases.json")
    corpus.cmd_backup(types.SimpleNamespace(name="uji"))
    assert (tmp_path / "backups" / "uji" / "processed" / "extraction_cache.json").exists()


def test_fact_embedding_cache_survives_backup_and_restore(tree, tmp_path, monkeypatch):
    _, processed = tree
    monkeypatch.setattr(corpus, "BACKUPS", tmp_path / "backups")
    monkeypatch.setattr(corpus, "EVAL_CASES", tmp_path / "no_eval_cases.json")
    monkeypatch.setattr(corpus, "_verify_after_restore", lambda: None)
    (processed / "fact_embeddings.npz").write_text("cache", encoding="utf-8")
    corpus.cmd_backup(types.SimpleNamespace(name="uji"))
    assert (tmp_path / "backups" / "uji" / "processed" / "fact_embeddings.npz").exists()

    corpus.cmd_clear(types.SimpleNamespace(yes=True))
    assert not (processed / "fact_embeddings.npz").exists()
    corpus.cmd_restore(types.SimpleNamespace(name="uji", yes=True))
    assert (processed / "fact_embeddings.npz").read_text(encoding="utf-8") == "cache"
