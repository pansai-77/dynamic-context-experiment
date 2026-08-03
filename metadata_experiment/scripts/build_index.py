import json

import pandas as pd

import _bootstrap  # noqa: F401

from metadata_experiment.config import settings
from metadata_experiment.index_metadata import (
    expected_manifest,
    manifest_path,
    verify_chunk_parity_with_exp1,
    write_index_manifest,
)
from metadata_experiment.topics import annotate_chunk, topic_names
from metadata_experiment.vector_store import MetadataVectorStore
from src.pdf_loader import chunk_pages, extract_pages


def main() -> None:
    pages = extract_pages(settings.book_dir, settings.book_file)
    chunks = chunk_pages(
        pages,
        target_size=settings.chunk_target_size,
        max_size=settings.chunk_max_size,
        min_size=settings.chunk_min_size,
        overlap=settings.chunk_overlap,
    )
    store = MetadataVectorStore(
        settings.qdrant_path, settings.collection_name, settings.embedding_model
    )
    store.rebuild(chunks)
    catalog = []
    for chunk in chunks:
        metadata = annotate_chunk(chunk)
        catalog.append({
            "Chunk ID": chunk.chunk_id,
            "Page Start": chunk.page_start,
            "Page End": chunk.page_end,
            "Characters": " | ".join(metadata["characters"]),
            "Topics": " | ".join(metadata["topics"]),
            "Keywords": " | ".join(metadata["keywords"]),
            "Importance": metadata["importance"],
            "Text": chunk.text,
        })
    catalog_path = settings.experiment_dir / "data" / "index_catalog.xlsx"
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(catalog).to_excel(catalog_path, index=False, sheet_name="Index Catalog")

    manifest = expected_manifest(settings, chunks, topic_names())
    output_manifest_path = manifest_path(settings.qdrant_path)
    write_index_manifest(output_manifest_path, manifest)

    print(f"Built {settings.collection_name} with {len(chunks)} chunks.")
    print(f"Manifest: {output_manifest_path}")
    print(f"Annotation catalog: {catalog_path}")

    try:
        parity_issues = verify_chunk_parity_with_exp1(settings)
    except FileNotFoundError as exc:
        print(f"Warning: skipped experiment 1 parity check ({exc}).")
        return

    if parity_issues:
        print("Warning: metadata index differs from experiment 1 index:")
        for issue in parity_issues:
            print(f"  - {issue}")
        return

    print("Experiment 1 chunk parity check passed.")


if __name__ == "__main__":
    main()
