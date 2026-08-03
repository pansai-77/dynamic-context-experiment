from __future__ import annotations

import csv
from pathlib import Path

from .metrics import parse_pipe_list


METHOD_A = "Query-Aware Top-4"
METHOD_B = "Query-Aware + Metadata Top-4"
METHODS = (METHOD_A, METHOD_B)


def should_retrieve(question_type: str) -> bool:
    return question_type.strip().lower() == "book"


def load_gold(path: Path) -> dict[str, dict[str, list[str]]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {
            str(row["Question ID"]): {
                "topics": parse_pipe_list(row.get("Gold Topics")),
                "chunks": parse_pipe_list(row.get("Gold Chunk IDs")),
            }
            for row in reader
        }
