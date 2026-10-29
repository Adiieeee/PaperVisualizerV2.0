"""corpus.py reset-index: delete the build index, keep the PDFs + paper_metadata.json. No network."""
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
                 "extraction_models.json", "extraction_cache.json", "fact_embeddings.npz", "external_cache.json",
                 ".gitkeep"):
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
        ".gitkeep", "external_cache.json", "extraction_cache.json", "paper_metadata.json"]


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
    assert not (processed / "external_cache.json").exists()


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


def test_external_cache_survives_backup_and_restore(tree, tmp_path, monkeypatch):
    _, processed = tree
    monkeypatch.setattr(corpus, "BACKUPS", tmp_path / "backups")
    monkeypatch.setattr(corpus, "EVAL_CASES", tmp_path / "no_eval_cases.json")
    monkeypatch.setattr(corpus, "_verify_after_restore", lambda: None)
    (processed / "external_cache.json").write_text("web", encoding="utf-8")
    corpus.cmd_backup(types.SimpleNamespace(name="uji"))
    corpus.cmd_clear(types.SimpleNamespace(yes=True))
    corpus.cmd_restore(types.SimpleNamespace(name="uji", yes=True))
    assert (processed / "external_cache.json").read_text(encoding="utf-8") == "web"


def test_external_forget_drops_one_entry_or_all(tmp_path, monkeypatch):
    from src import config
    from src.query.external import load_cache, save_cache
    path = str(tmp_path / "external_cache.json")
    monkeypatch.setattr(config, "EXTERNAL_CACHE_PATH", path)
    save_cache(path, {"MMLU": {"status": "ok"}, "Gemini": {"status": "ok"}})
    corpus.cmd_external_forget(types.SimpleNamespace(name="Gemini", all=False))
    assert list(load_cache(path)) == ["MMLU"]
    corpus.cmd_external_forget(types.SimpleNamespace(name="nope", all=False))   # unknown: nothing changes
    assert list(load_cache(path)) == ["MMLU"]
    corpus.cmd_external_forget(types.SimpleNamespace(name=None, all=True))
    assert load_cache(path) == {}
