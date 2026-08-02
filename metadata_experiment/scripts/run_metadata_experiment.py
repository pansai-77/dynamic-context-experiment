from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from config import settings
from experiment import (
    create_run_directory,
    export_run_config,
    format_detailed_dataframe,
    run_metadata_experiment,
)
from reporting import create_summary_workbook
from src.results_io import parse_question_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Metadata RAG Phase 1 experiment.")
    parser.add_argument("--method", action="append", default=None)
    parser.add_argument(
        "--question-ids",
        type=str,
        default=None,
        help="Comma-separated question IDs, e.g. Q01,Q02.",
    )
    parser.add_argument(
        "--with-fallback",
        action="store_true",
        help="Enable Top-2→Top-3/4 topic expansion for supplementary runs only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    question_ids = parse_question_ids(args.question_ids)
    methods = args.method

    run_dir = create_run_directory(settings.results_dir)
    detailed_path = run_dir / "detailed_results.xlsx"
    summary_path = run_dir / "summary_results.xlsx"
    run_config_path = run_dir / "run_config.json"

    detailed_df = run_metadata_experiment(
        settings,
        question_ids=question_ids,
        selected_methods=methods,
        allow_topic_expansion=args.with_fallback,
    )

    format_detailed_dataframe(detailed_df).to_excel(
        detailed_path,
        index=False,
        sheet_name="Detailed Results",
    )

    method_names = sorted(detailed_df["method"].unique().tolist())
    export_run_config(settings, method_names, run_config_path, run_dir, args.with_fallback)
    create_summary_workbook(detailed_path, summary_path)

    print(f"Run directory: {run_dir}")
    print(f"Detailed results: {detailed_path}")
    print(f"Summary results: {summary_path}")


if __name__ == "__main__":
    main()
