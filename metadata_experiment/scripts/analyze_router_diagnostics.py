from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from config import settings
from router_diagnostics import analyze_router_diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Router/filter diagnostics with Gold Retention Rate (read-only)."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write diagnostics JSON (table + summary).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df, summary = analyze_router_diagnostics(settings)
    print(df.to_string(index=False))
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.output:
        payload = {
            "summary": summary,
            "rows": df.to_dict(orient="records"),
        }
        args.output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"\nWrote diagnostics to {args.output}")


if __name__ == "__main__":
    main()
