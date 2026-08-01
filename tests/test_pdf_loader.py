from pathlib import Path

import pytest

from src.pdf_loader import chunk_pages, resolve_pdf_files


def test_chunk_pages_splits_by_characters():
    pages = [{"source_file": "test.pdf", "page_number": 1, "text": "字" * 600}]
    chunks = chunk_pages(pages, chunk_size=500, chunk_overlap=80)
    assert len(chunks) == 2
    assert len(chunks[0].text) == 500
    assert len(chunks[1].text) == 180


def test_resolve_pdf_files_uses_explicit_book_file(tmp_path: Path) -> None:
    book_dir = tmp_path / "data" / "book"
    book_dir.mkdir(parents=True)
    target = book_dir / "活着.pdf"
    target.write_bytes(b"%PDF-1.4")
    (book_dir / "other.pdf").write_bytes(b"%PDF-1.4")

    resolved = resolve_pdf_files(book_dir, target)
    assert resolved == [target]


def test_resolve_pdf_files_requires_existing_book_file(tmp_path: Path) -> None:
    book_dir = tmp_path / "data" / "book"
    book_dir.mkdir(parents=True)
    missing = book_dir / "missing.pdf"
    with pytest.raises(FileNotFoundError, match="Book file not found"):
        resolve_pdf_files(book_dir, missing)
