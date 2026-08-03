from __future__ import annotations

import math


def parse_pipe_list(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    return [part.strip() for part in str(value).split("|") if part.strip()]


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

