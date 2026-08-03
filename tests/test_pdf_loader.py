from pathlib import Path

import pytest

from src.pdf_loader import chunk_pages, resolve_pdf_files


def _sample_pages() -> list[dict]:
    return [
        {
            "source_file": "test.pdf",
            "page_number": 1,
            "text": "第一段。" + ("叙述" * 120) + "。",
        },
        {
            "source_file": "test.pdf",
            "page_number": 2,
            "text": "第二段开始。" + ("继续" * 130) + "。",
        },
    ]


def test_chunk_pages_uses_global_chunk_ids():
    chunks = chunk_pages(_sample_pages())
    assert chunks
    assert chunks[0].chunk_id == "c0001"
    assert all(chunk.chunk_id.startswith("c") for chunk in chunks)


def test_chunk_pages_can_span_pages():
    chunks = chunk_pages(_sample_pages(), target_size=200, max_size=300, min_size=50, overlap=50)
    cross_page = [chunk for chunk in chunks if chunk.page_start != chunk.page_end]
    assert cross_page


def test_chunk_pages_respects_max_size():
    pages = [{"source_file": "test.pdf", "page_number": 1, "text": "字" * 900}]
    chunks = chunk_pages(pages)
    assert all(len(chunk.text) <= 800 for chunk in chunks)


def test_chunk_pages_prefers_sentence_boundary():
    text = "甲。" + ("中间" * 120) + "乙。"
    pages = [{"source_file": "test.pdf", "page_number": 1, "text": text}]
    chunks = chunk_pages(pages, target_size=100, max_size=150, min_size=20, overlap=20)
    assert any(chunk.text.endswith("。") for chunk in chunks)


def test_chunk_pages_merges_short_tail():
    pages = [{"source_file": "test.pdf", "page_number": 1, "text": ("内容" * 200) + "尾"}]
    chunks = chunk_pages(pages, target_size=120, max_size=160, min_size=40, overlap=20)
    assert len(chunks) >= 1
    assert chunks[-1].text.endswith("尾")


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
