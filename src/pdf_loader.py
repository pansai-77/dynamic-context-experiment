from __future__ import annotations

import re
from pathlib import Path

import fitz

from .models import Chunk

SENTENCE_ENDS = set("。！？；")
PARAGRAPH_BREAK = "\n\n"


def _clean_text(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def resolve_pdf_files(book_dir: Path, book_file: Path | None = None) -> list[Path]:
    if book_file is not None:
        if not book_file.exists():
            raise FileNotFoundError(f"Book file not found: {book_file}")
        return [book_file]

    pdf_files = sorted(book_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {book_dir}")
    return pdf_files


def extract_pages(book_dir: Path, book_file: Path | None = None) -> list[dict]:
    pdf_files = resolve_pdf_files(book_dir, book_file)
    print("Indexing PDFs:", [path.name for path in pdf_files])

    pages = []
    global_page = 0
    for pdf_path in pdf_files:
        with fitz.open(pdf_path) as document:
            for local_page_index, page in enumerate(document):
                global_page += 1
                text = _clean_text(page.get_text("text"))
                if len(text) < 50:
                    continue
                pages.append(
                    {
                        "source_file": pdf_path.name,
                        "page_number": global_page,
                        "local_page_number": local_page_index + 1,
                        "text": text,
                    }
                )
    return pages


def _build_continuous_text(pages: list[dict]) -> tuple[str, list[tuple[int, int, int]]]:
    """Return full book text and (char_start, char_end, page_number) spans."""
    parts: list[str] = []
    spans: list[tuple[int, int, int]] = []
    offset = 0
    for index, page in enumerate(pages):
        if index > 0:
            parts.append("\n")
            offset += 1
        start = offset
        parts.append(page["text"])
        offset += len(page["text"])
        spans.append((start, offset, page["page_number"]))
    return "".join(parts), spans


def _page_range_for_span(
    start: int,
    end: int,
    page_spans: list[tuple[int, int, int]],
) -> tuple[int, int]:
    matched = [
        page_number
        for span_start, span_end, page_number in page_spans
        if span_start < end and span_end > start
    ]
    if not matched:
        return 0, 0
    return min(matched), max(matched)


def _is_paragraph_break(text: str, end: int) -> bool:
    return end >= 2 and text[end - 2 : end] == PARAGRAPH_BREAK


def _find_chunk_end(
    text: str,
    start: int,
    target_size: int,
    max_size: int,
    min_size: int,
) -> int:
    text_len = len(text)
    max_end = min(start + max_size, text_len)
    target_end = min(start + target_size, text_len)
    min_end = start + min_size

    if max_end <= start:
        return text_len
    if max_end - start <= min_size:
        return max_end

    for end in range(max_end, target_end - 1, -1):
        if _is_paragraph_break(text, end):
            return end

    for end in range(max_end, target_end - 1, -1):
        if text[end - 1] in SENTENCE_ENDS:
            return end

    for end in range(target_end, min_end, -1):
        if _is_paragraph_break(text, end):
            return end

    for end in range(target_end, min_end, -1):
        if text[end - 1] in SENTENCE_ENDS:
            return end

    return max_end


def chunk_pages(
    pages: list[dict],
    target_size: int = 600,
    max_size: int = 800,
    min_size: int = 100,
    overlap: int = 100,
) -> list[Chunk]:
    """Split continuous book text into overlapping, sentence/paragraph-aware chunks."""
    if not pages:
        return []
    if overlap >= max_size:
        raise ValueError("overlap must be smaller than max_size")
    if min_size > target_size or target_size > max_size:
        raise ValueError("expected min_size <= target_size <= max_size")

    full_text, page_spans = _build_continuous_text(pages)
    if not full_text.strip():
        return []

    source_file = pages[0]["source_file"]
    chunks: list[Chunk] = []
    start = 0
    chunk_index = 0

    while start < len(full_text):
        end = _find_chunk_end(full_text, start, target_size, max_size, min_size)
        fragment = full_text[start:end].strip()
        if not fragment:
            break

        page_start, page_end = _page_range_for_span(start, end, page_spans)

        if len(fragment) < min_size and chunks:
            previous = chunks[-1]
            merged_text = f"{previous.text}{fragment}"
            chunks[-1] = Chunk(
                chunk_id=previous.chunk_id,
                text=merged_text,
                source_file=previous.source_file,
                page_number=previous.page_start,
                page_start=previous.page_start,
                page_end=max(previous.page_end, page_end),
                chunk_index=previous.chunk_index,
            )
            break

        chunk_index += 1
        chunks.append(
            Chunk(
                chunk_id=f"c{chunk_index:04d}",
                text=fragment,
                source_file=source_file,
                page_number=page_start,
                page_start=page_start,
                page_end=page_end,
                chunk_index=chunk_index,
            )
        )

        if end >= len(full_text):
            break

        next_start = end - overlap
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks
