from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from config import settings
from experiment import merge_scoring_sheet
from reporting import create_summary_workbook


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge blind scores and regenerate summary.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Timestamped run directory under metadata_experiment/results/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    if not run_dir.is_absolute():
        run_dir = settings.results_dir / run_dir

    detailed_path = run_dir / "detailed_results.xlsx"
    scoring_path = run_dir / "scoring_sheet.xlsx"
    mapping_path = run_dir / "scoring_mapping.csv"
    summary_path = run_dir / "summary_results.xlsx"
    benchmark_path = run_dir / "retrieval_benchmark.xlsx"

    merged = merge_scoring_sheet(detailed_path, scoring_path, mapping_path)
    merged.to_excel(detailed_path, index=False, sheet_name="Detailed Results")
    create_summary_workbook(
        detailed_path,
        summary_path,
        benchmark_path if benchmark_path.exists() else None,
    )
    print(f"Updated detailed results: {detailed_path}")
    print(f"Updated summary results: {summary_path}")


if __name__ == "__main__":
    main()
