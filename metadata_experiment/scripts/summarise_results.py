from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from metadata_experiment.config import settings
from metadata_experiment.reporting import create_summary
from metadata_experiment.run_metadata import resolve_run_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate summary_results.xlsx from detailed_results.xlsx."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Timestamped run directory under metadata_experiment/results/. "
        "Defaults to the latest run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = resolve_run_directory(args.run_dir, settings.results_dir)
    detailed = run_dir / "detailed_results.xlsx"
    summary = run_dir / "summary_results.xlsx"
    if not detailed.exists():
        raise FileNotFoundError(f"Detailed results not found: {detailed}")

    create_summary(detailed, summary)
    print(f"Run directory: {run_dir}")
    print(f"Updated summary: {summary}")


if __name__ == "__main__":
    main()
