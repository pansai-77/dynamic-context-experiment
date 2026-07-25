from __future__ import annotations
import argparse
import _bootstrap  # noqa: F401

from src.config import settings
from src.experiment import export_detailed_results, resolve_methods, run_experiments
from src.reporting import create_summary_workbook
from src.run_metadata import create_run_directory, export_run_metadata

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--method", action="append", default=None)
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Include optional methods such as Query-Aware + Top-2.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    methods = resolve_methods(args.method, args.include_optional)
    run_dir = create_run_directory(settings.results_dir)
    df = run_experiments(settings, args.limit, args.method, args.include_optional)
    detailed = run_dir / "detailed_results.xlsx"
    summary = run_dir / "summary_results.xlsx"
    run_config = run_dir / "run_config.json"
    export_detailed_results(df, detailed)
    create_summary_workbook(detailed, summary)
    export_run_metadata(
        settings,
        [method.name for method in methods],
        run_config,
        run_directory=run_dir,
    )
    print(f"Run directory: {run_dir}")
    print(f"Detailed results: {detailed}")
    print(f"Summary results: {summary}")
    print(f"Run metadata: {run_config}")

if __name__ == "__main__":
    main()
