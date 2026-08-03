from __future__ import annotations

from src.pdf_loader import chunk_pages


def test_chunk_id_uses_global_sequence():
    pages = [
        {
            "page_number": 87,
            "text": "有庆" * 200,
            "source_file": "活着.pdf",
        }
    ]
    chunks = chunk_pages(pages)
    assert chunks
    assert chunks[0].chunk_id == "c0001"
    assert chunks[0].page_start == 87
    assert chunks[0].page_end == 87
