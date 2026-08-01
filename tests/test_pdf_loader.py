from src.pdf_loader import chunk_pages


def test_chunk_pages_splits_by_characters():
    pages = [{"source_file": "test.pdf", "page_number": 1, "text": "字" * 600}]
    chunks = chunk_pages(pages, chunk_size=500, chunk_overlap=80)
    assert len(chunks) == 2
    assert len(chunks[0].text) == 500
    assert len(chunks[1].text) == 180
