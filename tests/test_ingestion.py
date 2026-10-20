import fitz  # PyMuPDF
import pytest

from src.ingestion import document as D
from src.ingestion.document import DocumentProcessor


def _pdf(path, pages):
    doc = fitz.open()
    for i in range(pages):
        doc.new_page().insert_text((72, 72), f"Halaman {i} tentang ResNet dan ImageNet.")
    doc.save(str(path))
    doc.close()
    return str(path)


def test_normal_pdf_still_produces_chunks(tmp_path):
    chunks = DocumentProcessor().process_pdf(_pdf(tmp_path / "ok.pdf", 2))
    assert chunks and chunks[0]["doc_id"] == "ok.pdf" and "ResNet" in chunks[0]["text"]


def test_pdf_over_page_limit_is_skipped_not_fatal(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(D, "MAX_PDF_PAGES", 2)
    assert DocumentProcessor().process_pdf(_pdf(tmp_path / "big.pdf", 3)) == []
    assert "exceeds the 2-page limit" in capsys.readouterr().out


def test_pdf_over_size_limit_is_skipped_before_parsing(tmp_path, monkeypatch, capsys):
    p = _pdf(tmp_path / "s.pdf", 1)
    monkeypatch.setattr(D, "MAX_PDF_BYTES", 10)
    assert DocumentProcessor().process_pdf(p) == []
    assert "exceeds" in capsys.readouterr().out


def test_one_oversized_pdf_does_not_abort_the_directory(tmp_path, monkeypatch):
    _pdf(tmp_path / "a.pdf", 1)
    _pdf(tmp_path / "huge.pdf", 5)
    monkeypatch.setattr(D, "MAX_PDF_PAGES", 3)
    chunks = DocumentProcessor().process_directory(str(tmp_path))
    assert {c["doc_id"] for c in chunks} == {"a.pdf"}


def test_corpus_backup_names_reject_path_traversal():
    import corpus
    for bad in ("../x", "..", "a/b", "a\\b", ".hidden", "", "x..y", "C:evil"):
        assert not corpus._valid_name(bad), bad
    for good in ("cnn-klasik", "nlp_transformer", "v1.2", "Tema-2026"):
        assert corpus._valid_name(good), good
