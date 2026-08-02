from __future__ import annotations

from src.pdf_loader import chunk_pages


def test_chunk_id_matches_experiment_one_format():
    pages = [
        {
            "page_number": 87,
            "text": "有庆" * 200,
            "source_file": "活着.pdf",
        }
    ]
    chunks = chunk_pages(pages, chunk_size=500, chunk_overlap=80)
    assert chunks
    assert chunks[0].chunk_id.startswith("p087-c")
    assert chunks[0].chunk_id == "p087-c001"
