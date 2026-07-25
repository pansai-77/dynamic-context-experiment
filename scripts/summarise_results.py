import argparse
import _bootstrap  # noqa: F401

from pathlib import Path

from src.config import settings
from src.reporting import create_summary_workbook
from src.run_metadata import find_latest_run_directory, is_run_directory

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Timestamped run directory under results/. Defaults to the latest run.",
    )
    return parser.parse_args()

def resolve_run_directory(run_dir: Path | None) -> Path:
    if run_dir is None:
        return find_latest_run_directory(settings.results_dir)
    if not run_dir.is_absolute():
        run_dir = settings.results_dir / run_dir
    if not is_run_directory(run_dir):
        raise ValueError(f"Not a timestamped run directory: {run_dir}")
    return run_dir

def main() -> None:
    args = parse_args()
    run_dir = resolve_run_directory(args.run_dir)
    detailed = run_dir / "detailed_results.xlsx"
    summary = run_dir / "summary_results.xlsx"
    if not detailed.exists():
        raise FileNotFoundError(f"Detailed results not found: {detailed}")
    create_summary_workbook(detailed, summary)
    print(f"Run directory: {run_dir}")
    print(f"Updated summary: {summary}")

if __name__ == "__main__":
    main()
