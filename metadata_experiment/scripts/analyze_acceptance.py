from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from acceptance_analysis import analyze_acceptance_report, format_confusion_table
from config import settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze metadata acceptance JSON with confusion matrix.")
    parser.add_argument(
        "report",
        type=Path,
        nargs="?",
        default=None,
        help="Path to acceptance_*.json (default: latest in results dir).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write analysis JSON.",
    )
    return parser.parse_args()


def _latest_acceptance_report(results_dir: Path) -> Path:
    candidates = sorted(results_dir.glob("acceptance_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No acceptance_*.json found under {results_dir}")
    return candidates[-1]


def main() -> None:
    args = parse_args()
    report_path = args.report or _latest_acceptance_report(settings.results_dir)
    analysis = analyze_acceptance_report(report_path)

    print(format_confusion_table(analysis))
    print()
    print(
        f"Primary pass: {analysis['primary_pass_count']}/{analysis['total_samples']} "
        f"({analysis['primary_pass_rate']:.1%})"
    )
    print(
        f"Topic-set pass: {analysis['topic_set_pass_count']}/{analysis['total_samples']} "
        f"({analysis['topic_set_pass_rate']:.1%})"
    )
    print(f"war as primary: {analysis['war_as_primary']}/{analysis['total_samples']} ({analysis['war_rate']:.1%})")
    print(f"family as primary: {analysis['family_as_primary']}")
    print(f"livelihood as primary: {analysis['livelihood_as_primary']}")
    print()
    print("Per category (primary / topic-set):")
    for category_id, bucket in analysis["by_category"].items():
        print(
            f"  {category_id}: {bucket['primary_pass']}/{bucket['total']} primary, "
            f"{bucket['topic_set_pass']}/{bucket['total']} topic-set"
        )

    if args.output:
        args.output.write_text(json.dumps(analysis, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nWrote analysis to {args.output}")


if __name__ == "__main__":
    main()
