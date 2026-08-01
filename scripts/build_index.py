import _bootstrap  # noqa: F401

from src.config import settings
from src.index_metadata import expected_metadata, write_index_metadata
from src.pdf_loader import chunk_pages, extract_pages, resolve_pdf_files
from src.vector_store import LocalVectorStore

def main() -> None:
    pdf_files = resolve_pdf_files(settings.book_dir, settings.book_file)
    pages = extract_pages(settings.book_dir, settings.book_file)
    chunks = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)
    print(f"Extracted {len(pages)} pages; created {len(chunks)} chunks.")
    store = LocalVectorStore(
        settings.qdrant_path, settings.collection_name, settings.embedding_model
    )
    store.rebuild(chunks)
    metadata = expected_metadata(settings, [path.name for path in pdf_files])
    write_index_metadata(settings.qdrant_path, metadata)
    print(f"Wrote index metadata to {settings.qdrant_path / 'index_metadata.json'}")
    print("Qdrant index built successfully.")

if __name__ == "__main__":
    main()
