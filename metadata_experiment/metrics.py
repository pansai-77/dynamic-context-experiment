from __future__ import annotations

import csv
import math
from pathlib import Path

METHOD_A = "Query-Aware Top-4"
METHOD_B = "Query-Aware + Metadata Top-4"
METHODS = (METHOD_A, METHOD_B)


def should_retrieve(question_type: str) -> bool:
    return question_type.strip().lower() == "book"


def parse_pipe_list(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    return [part.strip() for part in str(value).split("|") if part.strip()]


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


def ranking_metrics(retrieved_ids: list[str], gold_ids: list[str]) -> tuple[float | None, float | None]:
    if not gold_ids:
        return None, None
    gold = set(gold_ids)
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in gold:
            return 1.0, 1.0 / rank
    return 0.0, 0.0


def filter_accuracy(routed_topics: list[str], gold_topics: list[str]) -> float | None:
    if not gold_topics:
        return None
    return float(bool(set(routed_topics) & set(gold_topics)))
