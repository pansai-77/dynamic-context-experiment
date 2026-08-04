import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from metadata_experiment.classification import ChunkClassificationResult, create_topic_classifier
from metadata_experiment.classification_prompts import (
    build_chunk_classification_prompt_for_text,
    classification_prompt_metadata,
)
from metadata_experiment.config import settings
from metadata_experiment.index_builder import load_chunks_from_catalog, seed_everything
from metadata_experiment.metadata_quality import (
    DIAGNOSTIC_CHUNK_IDS,
    audit_content_warnings,
    evaluate_diagnostic_expectations,
)
from src.models import Chunk
from src.pdf_loader import chunk_pages, extract_pages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run development-set metadata diagnostics.")
    parser.add_argument(
        "--split",
        choices=("development", "validation"),
        default="development",
        help="Chunk split to evaluate. Validation set must be annotated before use.",
    )
    parser.add_argument(
        "--chunk-ids",
        nargs="*",
        help="Optional explicit chunk IDs. Defaults to the selected split.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass classification cache and always call the LLM.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompts and chunk text only; do not call the LLM.",
    )
    return parser.parse_args()


def load_chunk_splits(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {
            "development": list(DIAGNOSTIC_CHUNK_IDS),
            "validation": [],
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "development": list(payload.get("development", [])),
        "validation": list(payload.get("validation", [])),
    }


def load_chunks(use_catalog: bool):
    if use_catalog:
        catalog_path = settings.experiment_dir / "data" / "index_catalog.xlsx"
        return load_chunks_from_catalog(catalog_path)
    pages = extract_pages(settings.book_dir, settings.book_file)
    return chunk_pages(
        pages,
        target_size=settings.chunk_target_size,
        max_size=settings.chunk_max_size,
        min_size=settings.chunk_min_size,
        overlap=settings.chunk_overlap,
    )


def print_classification_diagnostics(result: ChunkClassificationResult, chunk: Chunk) -> None:
    metadata = classification_prompt_metadata()
    print("=" * 80)
    print(f"Chunk: {chunk.chunk_id} | cache_hit={result.cache_hit}")
    print(f"Cache version: {result.cache_version or 'disabled'}")
    print()
    print("1) Allowed topics:")
    print(f"   count={metadata['allowed_topic_count']}")
    print()
    print("2) Final classification prompt:")
    print(result.prompt)
    print()
    print("3) raw_output:")
    print(result.raw_response or "<empty>")
    print()
    print("4) parsed_topics:")
    print(result.parsed_topics)
    print()
    print("5) validation_warnings:")
    print(list(result.validation_warnings) or "-")
    print()
    print("6) final_topics:")
    print(result.final_topics)
    print()
    dev_notes = evaluate_diagnostic_expectations(chunk.chunk_id, result.final_topics)
    if dev_notes:
        print("7) development expectations (informational only):")
        for note in dev_notes:
            print(f"   - {note}")
        print()
    preview = chunk.text.replace("\n", " ")
    if len(preview) > 240:
        preview = preview[:240] + "..."
    print(f"Chunk text preview: {preview}")


def main() -> None:
    args = parse_args()
    seed_everything(settings.random_seed)

    splits_path = settings.experiment_dir / "data" / "chunk_splits.json"
    splits = load_chunk_splits(splits_path)
    chunk_ids = args.chunk_ids or splits[args.split]
    if args.split == "validation" and not chunk_ids:
        raise SystemExit(
            "Validation split is empty. Annotate 20-30 unseen chunks in "
            "metadata_experiment/data/chunk_splits.json before use."
        )

    chunks = load_chunks(args.dry_run)
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    missing = [chunk_id for chunk_id in chunk_ids if chunk_id not in chunk_by_id]
    if missing:
        raise SystemExit(f"Unknown chunk IDs: {', '.join(missing)}")

    if args.dry_run:
        for chunk_id in chunk_ids:
            chunk = chunk_by_id[chunk_id]
            prompt = build_chunk_classification_prompt_for_text(chunk.text)
            result = ChunkClassificationResult(
                chunk_id=chunk_id,
                raw_response="<dry-run>",
                parsed_topics=[],
                validation_warnings=tuple(),
                final_topics=[],
                attempts=0,
                prompt=prompt,
            )
            print_classification_diagnostics(result, chunk)
        return

    classifier = create_topic_classifier(settings)

    for chunk_id in chunk_ids:
        chunk = chunk_by_id[chunk_id]
        result = classifier.classify(chunk, use_cache=not args.no_cache)
        print_classification_diagnostics(result, chunk)
        for warning in audit_content_warnings(chunk.text, result.final_topics):
            print(f"AUDIT WARNING: {warning}")


if __name__ == "__main__":
    main()
