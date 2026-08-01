from __future__ import annotations
from pathlib import Path
import re
import fitz
from .models import Chunk

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
                pages.append({
                    "source_file": pdf_path.name,
                    "page_number": global_page,
                    "local_page_number": local_page_index + 1,
                    "text": text,
                })
    return pages

def chunk_pages(pages: list[dict], chunk_size: int = 500, chunk_overlap: int = 80) -> list[Chunk]:
    """Split page text into overlapping chunks by character count (for Chinese prose)."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    chunks = []
    step = chunk_size - chunk_overlap
    for page in pages:
        text = page["text"]
        page_chunk_index = 0
        for start in range(0, len(text), step):
            fragment = text[start:start + chunk_size].strip()
            if not fragment:
                break
            page_chunk_index += 1
            chunks.append(Chunk(
                chunk_id=f"p{page['page_number']:03d}-c{page_chunk_index:03d}",
                text=fragment,
                source_file=page["source_file"],
                page_number=page["page_number"],
                chunk_index=page_chunk_index,
            ))
            if start + chunk_size >= len(text):
                break
    return chunks
