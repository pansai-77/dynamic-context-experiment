from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from metadata_experiment.config import settings
from metadata_experiment.experiment import METHODS, run_experiment
from metadata_experiment.reporting import create_summary, export_detailed
from metadata_experiment.run_metadata import create_run_directory, export_run_metadata


TIMING_NOTES = [
    "Method B performs query embedding twice: once in Router Time and once in Vector Time.",
    "Retrieval Time = Router Time + Vector Time; do not compare Vector Time alone across methods.",
    "Compare end-to-end Retrieval Time when reporting latency impact of metadata filtering.",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--question-ids", type=str)
    args = parser.parse_args()
    question_ids = [x.strip() for x in args.question_ids.split(",")] if args.question_ids else None

    run_dir = create_run_directory(settings.results_dir)
    detailed_path = run_dir / "detailed_results.xlsx"
    summary_path = run_dir / "summary_results.xlsx"
    run_config_path = run_dir / "run_config.json"

    frame = run_experiment(settings, args.limit, question_ids)
    export_detailed(frame, detailed_path)
    create_summary(detailed_path, summary_path)
    export_run_metadata(
        settings,
        list(METHODS),
        run_config_path,
        run_directory=run_dir,
        notes=TIMING_NOTES,
    )

    print(f"Run directory: {run_dir}")
    print(f"Detailed results: {detailed_path}")
    print(f"Summary results: {summary_path}")
    print(f"Run metadata: {run_config_path}")


if __name__ == "__main__":
    main()
