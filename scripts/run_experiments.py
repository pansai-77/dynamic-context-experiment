from __future__ import annotations
import argparse
from pathlib import Path
import _bootstrap  # noqa: F401

from src.config import settings
from src.experiment import export_detailed_results, resolve_methods, run_experiments
from src.results_io import parse_question_ids, patch_detailed_results
from src.reporting import create_summary_workbook
from src.run_metadata import (
    create_run_directory,
    export_run_metadata,
    record_partial_rerun,
    resolve_run_directory,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--method", action="append", default=None)
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Include optional methods such as Query-Aware + Top-2.",
    )
    parser.add_argument(
        "--question-ids",
        type=str,
        default=None,
        help="Comma-separated question IDs to run, e.g. Q16,Q17,Q18,Q19,Q20.",
    )
    parser.add_argument(
        "--patch-run-dir",
        type=Path,
        default=None,
        help="Merge results into an existing timestamped run directory instead of creating a new one.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    question_ids = parse_question_ids(args.question_ids)
    methods = resolve_methods(args.method, args.include_optional)
    method_names = [method.name for method in methods]

    if args.patch_run_dir and not question_ids:
        raise SystemExit("--patch-run-dir requires --question-ids.")

    patching = args.patch_run_dir is not None
    if patching:
        run_dir = resolve_run_directory(args.patch_run_dir, settings.results_dir)
        detailed = run_dir / "detailed_results.xlsx"
        if not detailed.exists():
            raise FileNotFoundError(f"Detailed results not found: {detailed}")
    else:
        run_dir = create_run_directory(settings.results_dir)
        detailed = run_dir / "detailed_results.xlsx"

    summary = run_dir / "summary_results.xlsx"
    run_config = run_dir / "run_config.json"

    df = run_experiments(
        settings,
        args.limit,
        args.method,
        args.include_optional,
        question_ids,
    )

    if patching:
        patched = patch_detailed_results(detailed, df)
        patched.to_excel(detailed, index=False, sheet_name="Detailed Results")
        record_partial_rerun(run_config, question_ids, method_names)
        print(f"Patched {len(df)} row(s) into existing run.")
    else:
        export_detailed_results(df, detailed)
        export_run_metadata(
            settings,
            method_names,
            run_config,
            run_directory=run_dir,
        )

    create_summary_workbook(detailed, summary)
    print(f"Run directory: {run_dir}")
    print(f"Detailed results: {detailed}")
    print(f"Summary results: {summary}")
    if not patching:
        print(f"Run metadata: {run_config}")

if __name__ == "__main__":
    main()
