from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401

from config import settings
from experiment import (
    create_run_directory,
    export_run_config,
    format_detailed_dataframe,
    run_metadata_experiment,
    summarize_benchmark,
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
        "--skip-benchmark",
        action="store_true",
        help="Skip the 25x retrieval benchmark (debug only).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    question_ids = parse_question_ids(args.question_ids)
    methods = args.method

    run_dir = create_run_directory(settings.results_dir)
    detailed_path = run_dir / "detailed_results.xlsx"
    benchmark_path = run_dir / "retrieval_benchmark.xlsx"
    benchmark_summary_path = run_dir / "retrieval_benchmark_summary.xlsx"
    scoring_path = run_dir / "scoring_sheet.xlsx"
    mapping_path = run_dir / "scoring_mapping.csv"
    summary_path = run_dir / "summary_results.xlsx"
    run_config_path = run_dir / "run_config.json"

    detailed_df, benchmark_df, scoring_df, mapping_df = run_metadata_experiment(
        settings,
        question_ids=question_ids,
        selected_methods=methods,
        skip_benchmark=args.skip_benchmark,
    )

    format_detailed_dataframe(detailed_df).to_excel(
        detailed_path,
        index=False,
        sheet_name="Detailed Results",
    )
    if not benchmark_df.empty:
        benchmark_summary = summarize_benchmark(benchmark_df)
        with pd.ExcelWriter(benchmark_path, engine="openpyxl") as writer:
            benchmark_df.to_excel(writer, index=False, sheet_name="Benchmark Runs")
            benchmark_summary.to_excel(writer, index=False, sheet_name="Benchmark Summary")
        benchmark_summary.to_excel(benchmark_summary_path, index=False, sheet_name="Benchmark Summary")
    else:
        benchmark_summary_path = None

    scoring_df.to_excel(scoring_path, index=False, sheet_name="Scoring")
    mapping_df.to_csv(mapping_path, index=False)

    method_names = sorted(detailed_df["method"].unique().tolist())
    export_run_config(settings, method_names, run_config_path, run_dir)
    create_summary_workbook(
        detailed_path,
        summary_path,
        benchmark_summary_path,
    )

    print(f"Run directory: {run_dir}")
    print(f"Detailed results: {detailed_path}")
    print(f"Scoring sheet: {scoring_path}")
    print(f"Summary results: {summary_path}")


if __name__ == "__main__":
    main()
