from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from config import settings
from reporting import create_summary_workbook, migrate_detailed_workbook


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate detailed results and regenerate summary.")
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
    summary_path = run_dir / "summary_results.xlsx"

    migrate_detailed_workbook(detailed_path)
    create_summary_workbook(detailed_path, summary_path)
    print(f"Updated detailed results: {detailed_path}")
    print(f"Updated summary results: {summary_path}")


if __name__ == "__main__":
    main()
