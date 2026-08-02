from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from config import settings
from filter_stats import analyze_book_filter_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze per-question metadata filter stats for Book questions."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write filter stats as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = analyze_book_filter_stats(settings)
    print(df.to_string(index=False))
    if args.output:
        payload = df.to_dict(orient="records")
        args.output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"\nWrote filter stats to {args.output}")


if __name__ == "__main__":
    main()
