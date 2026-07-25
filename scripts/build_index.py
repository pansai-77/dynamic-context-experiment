from src.config import settings
from src.pdf_loader import chunk_pages, extract_pages
from src.vector_store import LocalVectorStore

def main() -> None:
    pages = extract_pages(settings.book_dir)
    chunks = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)
    print(f"Extracted {len(pages)} pages; created {len(chunks)} chunks.")
    store = LocalVectorStore(
        settings.qdrant_path, settings.collection_name, settings.embedding_model
    )
    store.rebuild(chunks)
    print("Qdrant index built successfully.")

if __name__ == "__main__":
    main()
