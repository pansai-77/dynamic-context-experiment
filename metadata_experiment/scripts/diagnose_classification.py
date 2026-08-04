import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import _bootstrap  # noqa: F401

from metadata_experiment.classification import (
    ChunkClassificationError,
    ChunkClassificationResult,
    create_topic_classifier,
)
from metadata_experiment.classification_prompts import (
    build_chunk_classification_prompt_for_text,
    classification_prompt_metadata,
)
from metadata_experiment.config import settings
from metadata_experiment.index_builder import load_chunks_from_catalog, seed_everything
from metadata_experiment.metadata_quality import (
    DIAGNOSTIC_CHUNK_IDS,
    evaluate_diagnostic_expectations,
)
from src.models import Chunk
from src.pdf_loader import chunk_pages, extract_pages


@dataclass
class DiagnosticRecord:
    chunk_id: str
    status: str
    soft_warnings: list[str] = field(default_factory=list)
    hard_mismatches: list[str] = field(default_factory=list)
    result: ChunkClassificationResult | None = None
    error: str = ""


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


def resolve_status(
    *,
    soft_warnings: list[str],
    hard_mismatches: list[str],
) -> str:
    if hard_mismatches:
        return "hard mismatch"
    if soft_warnings:
        return "soft warning"
    return "success"


def print_classification_diagnostics(
    result: ChunkClassificationResult,
    chunk: Chunk,
    *,
    status: str,
    soft_warnings: list[str],
    hard_mismatches: list[str],
) -> None:
    metadata = classification_prompt_metadata()
    print("=" * 80)
    print(f"Chunk: {chunk.chunk_id} | status={status} | cache_hit={result.cache_hit}")
    print(f"Prompt version: {metadata['prompt_version']} | few_shot={metadata['few_shot']}")
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
    print(soft_warnings or "-")
    print()
    print("6) final_topics:")
    print(result.final_topics)
    print()
    if hard_mismatches:
        print("7) development mismatch (informational only):")
        for note in hard_mismatches:
            print(f"   - {note}")
        print()
    preview = chunk.text.replace("\n", " ")
    if len(preview) > 240:
        preview = preview[:240] + "..."
    print(f"Chunk text preview: {preview}")


def print_failure_diagnostics(chunk: Chunk, exc: ChunkClassificationError) -> None:
    print("=" * 80)
    print(f"Chunk: {chunk.chunk_id} | status=failure")
    print(f"Error: {exc}")
    if exc.last_response:
        print()
        print("Last raw response:")
        print(exc.last_response)
    preview = chunk.text.replace("\n", " ")
    if len(preview) > 240:
        preview = preview[:240] + "..."
    print()
    print(f"Chunk text preview: {preview}")


def classify_chunk_record(
    chunk: Chunk,
    classifier,
    *,
    use_cache: bool,
) -> DiagnosticRecord:
    try:
        result = classifier.classify(chunk, use_cache=use_cache)
    except ChunkClassificationError as exc:
        print_failure_diagnostics(chunk, exc)
        return DiagnosticRecord(
            chunk_id=chunk.chunk_id,
            status="failure",
            error=str(exc),
        )

    soft_warnings = list(result.validation_warnings)
    hard_mismatches = evaluate_diagnostic_expectations(chunk.chunk_id, result.final_topics)
    status = resolve_status(
        soft_warnings=soft_warnings,
        hard_mismatches=hard_mismatches,
    )
    print_classification_diagnostics(
        result,
        chunk,
        status=status,
        soft_warnings=soft_warnings,
        hard_mismatches=hard_mismatches,
    )
    for warning in soft_warnings:
        print(f"SOFT WARNING: {warning}")

    return DiagnosticRecord(
        chunk_id=chunk.chunk_id,
        status=status,
        soft_warnings=soft_warnings,
        hard_mismatches=hard_mismatches,
        result=result,
    )


def topic_distribution(records: list[DiagnosticRecord]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        if record.result is None:
            continue
        counts.update(record.result.final_topics)
    return counts


def count_medical_topic(records: list[DiagnosticRecord]) -> int:
    return sum(
        1
        for record in records
        if record.result and "医疗献血" in record.result.final_topics
    )


def count_broad_topic_warnings(records: list[DiagnosticRecord]) -> int:
    return sum(
        1
        for record in records
        if any("only broad topics assigned" in warning for warning in record.soft_warnings)
    )


def print_summary(records: list[DiagnosticRecord]) -> None:
    success = [record for record in records if record.status == "success"]
    soft_warning = [record for record in records if record.status == "soft warning"]
    hard_mismatch = [record for record in records if record.status == "hard mismatch"]
    failure = [record for record in records if record.status == "failure"]
    distribution = topic_distribution(records)

    print()
    print("=" * 80)
    print("Diagnostic summary")
    print("=" * 80)
    print(f"Total: {len(records)}")
    print(f"Success: {len(success)}")
    print(f"Soft warning: {len(soft_warning)}")
    print(f"Hard mismatch: {len(hard_mismatch)}")
    print(f"Failure: {len(failure)}")
    print(f"Illegal Topic failures: {sum(1 for record in failure if 'illegal topic:' in record.error)}")
    print(f"医疗献血 assigned: {count_medical_topic(records)}")
    print(f"Broad topic warnings: {count_broad_topic_warnings(records)}")
    print(f"Development mismatch count: {len(hard_mismatch)}")
    print()
    print("Topic distribution:")
    for topic, count in sorted(distribution.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {topic}: {count}")
    print()

    for label, group in (
        ("Success", success),
        ("Soft warning", soft_warning),
        ("Hard mismatch", hard_mismatch),
        ("Failure list", failure),
    ):
        if not group:
            continue
        print(f"{label}:")
        for record in group:
            if record.result:
                print(f"  - {record.chunk_id}: {record.result.final_topics}")
                for warning in record.soft_warnings:
                    print(f"      soft: {warning}")
                for note in record.hard_mismatches:
                    print(f"      hard: {note}")
            else:
                print(f"  - {record.chunk_id}: {record.error}")
        print()


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
            print_classification_diagnostics(
                result,
                chunk,
                status="success",
                soft_warnings=[],
                hard_mismatches=evaluate_diagnostic_expectations(chunk_id, []),
            )
        return

    classifier = create_topic_classifier(settings)
    records: list[DiagnosticRecord] = []

    for chunk_id in chunk_ids:
        chunk = chunk_by_id[chunk_id]
        records.append(classify_chunk_record(
            chunk,
            classifier,
            use_cache=not args.no_cache,
        ))

    print_summary(records)


if __name__ == "__main__":
    main()
