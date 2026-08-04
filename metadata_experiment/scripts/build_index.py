import argparse

import _bootstrap  # noqa: F401

from metadata_experiment.classification import create_topic_classifier
from metadata_experiment.config import settings
from metadata_experiment.index_builder import (
    classify_chunks,
    export_classification_audit,
    export_original_metadata,
    print_pre_build_quality_report,
    run_full_index_build,
    seed_everything,
)
from metadata_experiment.manual_review import (
    ManualReviewError,
    export_classification_failures,
    export_manual_override_template,
    format_manual_review_summary,
    load_manual_topic_overrides,
    merge_topics_with_manual_overrides,
)
from metadata_experiment.metadata_quality import (
    QualitySampleRecord,
    format_quality_sample_report,
    load_or_create_quality_sample,
)
from src.pdf_loader import chunk_pages, extract_pages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit-chunks",
        type=int,
        help="Process only the first N chunks. Without --write-index, classify only.",
    )
    parser.add_argument(
        "--write-index",
        action="store_true",
        help="After metadata generation, write Qdrant index, catalog, and manifest.",
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Keep classifying after LLM failures and export failure/manual-review CSV files.",
    )
    parser.add_argument(
        "--use-manual-overrides",
        action="store_true",
        help="Apply manual_topic_overrides.csv to failed chunks before writing the index.",
    )
    parser.add_argument(
        "--quality-sample",
        action="store_true",
        help="Run fixed-seed quality sample check excluding development diagnostic chunks.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass classification cache during LLM classification.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print chunk text, raw LLM JSON, and final topics for each chunk.",
    )
    return parser.parse_args()


def run_quality_sample(*, use_cache: bool) -> None:
    pages = extract_pages(settings.book_dir, settings.book_file)
    chunks = chunk_pages(
        pages,
        target_size=settings.chunk_target_size,
        max_size=settings.chunk_max_size,
        min_size=settings.chunk_min_size,
        overlap=settings.chunk_overlap,
    )
    sample_ids = load_or_create_quality_sample(
        all_chunk_ids=[chunk.chunk_id for chunk in chunks],
        sample_file=settings.quality_sample_file,
        sample_size=settings.quality_sample_size,
        seed=settings.random_seed,
    )
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    classifier = create_topic_classifier(settings)

    records: list[QualitySampleRecord] = []
    for chunk_id in sample_ids:
        chunk = chunk_by_id[chunk_id]
        result = classifier.classify(chunk, use_cache=use_cache)
        records.append(QualitySampleRecord(
            chunk_id=chunk_id,
            topics=list(result.final_topics),
            validation_warnings=list(result.validation_warnings),
        ))

    print(format_quality_sample_report(records))
    print()
    print(f"Fixed sample saved at: {settings.quality_sample_file}")
    print("Manual review: verify each topic assignment is basically reasonable before full build.")


def main() -> None:
    args = parse_args()
    seed_everything(settings.random_seed)

    if args.quality_sample:
        run_quality_sample(use_cache=not args.no_cache)
        return

    if args.write_index and not args.use_manual_overrides:
        raise SystemExit(
            "Full index writes require --use-manual-overrides after reviewing "
            f"{settings.manual_topic_overrides_file.name}."
        )

    if args.write_index and not args.limit_chunks:
        print("Building full metadata index with manual review overrides.")
    if args.limit_chunks is not None and args.limit_chunks <= 0:
        raise SystemExit("--limit-chunks must be a positive integer.")

    pages = extract_pages(settings.book_dir, settings.book_file)
    chunks = chunk_pages(
        pages,
        target_size=settings.chunk_target_size,
        max_size=settings.chunk_max_size,
        min_size=settings.chunk_min_size,
        overlap=settings.chunk_overlap,
    )
    if args.limit_chunks:
        chunks = chunks[: args.limit_chunks]

    continue_on_failure = args.continue_on_failure or args.use_manual_overrides
    classifier = create_topic_classifier(settings)
    results, failures = classify_chunks(
        chunks,
        classifier,
        show_details=args.verbose or (args.limit_chunks is not None and args.limit_chunks <= 3),
        use_cache=not args.no_cache,
        continue_on_failure=continue_on_failure,
    )

    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    manual_overrides = (
        load_manual_topic_overrides(settings.manual_topic_overrides_file)
        if args.use_manual_overrides
        else {}
    )

    if failures:
        export_classification_failures(
            settings.classification_failures_file,
            failures,
            text_by_chunk_id={chunk_id: chunk.text for chunk_id, chunk in chunk_by_id.items()},
        )
        export_manual_override_template(
            settings.manual_topic_overrides_template_file,
            failures,
        )
        print()
        print(f"Classification failures exported to: {settings.classification_failures_file}")
        print(
            "Manual override template exported to: "
            f"{settings.manual_topic_overrides_template_file}"
        )
        print(
            "Fill topics in "
            f"{settings.manual_topic_overrides_file} "
            "(copy rows from the template file if needed)."
        )

    if args.use_manual_overrides:
        try:
            indexed_topics, sources = merge_topics_with_manual_overrides(
                results,
                failures,
                manual_overrides,
                all_chunk_ids=[chunk.chunk_id for chunk in chunks],
            )
        except ManualReviewError as exc:
            raise SystemExit(str(exc)) from exc
    else:
        indexed_topics = {result.chunk_id: list(result.final_topics) for result in results}
        sources = {result.chunk_id: "llm" for result in results}
        if failures:
            print()
            print(format_manual_review_summary(
                results=results,
                failures=failures,
                manual_overrides=manual_overrides,
            ))
            print()
            print(
                "Next step: fill topics in "
                f"{settings.manual_topic_overrides_file}, then rerun with "
                "--write-index --use-manual-overrides."
            )

    export_original_metadata(
        settings.original_metadata_file,
        topics_by_chunk_id=indexed_topics,
        raw_outputs={result.chunk_id: result.raw_response for result in results},
        sources=sources,
    )
    print_pre_build_quality_report(results, settings, indexed_topics=indexed_topics)
    export_classification_audit(
        results,
        settings.experiment_dir / "data" / "classification_audit.json",
    )

    if args.use_manual_overrides:
        print()
        print(format_manual_review_summary(
            results=results,
            failures=failures,
            manual_overrides=manual_overrides,
            topics_by_chunk_id=indexed_topics,
            source_by_chunk_id=sources,
        ))

    if args.limit_chunks and not args.write_index:
        print()
        print("Metadata generation test complete.")
        print(f"Original metadata saved to: {settings.original_metadata_file}")
        return

    if not args.write_index:
        if not failures:
            print()
            print("LLM classification completed without failures.")
            print(
                "Review original_metadata.csv, then rerun with "
                "--write-index --use-manual-overrides if you still want manual review."
            )
        return

    catalog_path = settings.experiment_dir / "data" / "index_catalog.xlsx"
    run_full_index_build(
        settings,
        chunks,
        indexed_topics,
        catalog_path,
        skip_parity_check=args.limit_chunks is not None,
    )


if __name__ == "__main__":
    main()
