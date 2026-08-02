from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def _normalize_topics(topics: list[str]) -> list[str]:
    return topics[:1]


def _primary_label(topics: list[str]) -> str:
    normalized = _normalize_topics(topics)
    if not normalized:
        return "(empty)"
    return normalized[0]


def _primary_hint(sample: dict) -> str:
    hint = sample.get("category_hint", "")
    if hint:
        return hint
    acceptable = sample.get("acceptable_topics") or sample.get("expected_topics") or []
    if not acceptable:
        return "(none)"
    first = acceptable[0]
    if isinstance(first, list):
        return _primary_label(first)
    return str(first)


def is_acceptable(actual_topics: list[str], acceptable_options: list[list[str]]) -> bool:
    actual = _normalize_topics(actual_topics)
    for option in acceptable_options:
        if actual == option:
            return True
    return False


def _resolve_acceptable_options(sample: dict) -> list[list[str]]:
    if "acceptable_topics" in sample:
        return sample["acceptable_topics"]
    expected = sample.get("expected_topics") or []
    return [expected] if expected else [[]]


def analyze_acceptance_report(report_path: Path) -> dict:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    samples = payload["samples"]

    by_category: dict[str, dict] = {}
    confusion: Counter[tuple[str, str]] = Counter()
    topic_predicted = Counter[str]()

    strict_pass = 0
    lenient_pass = 0

    for row in samples:
        category = row.get("category_hint") or row.get("category_id", "unknown")
        acceptable_options = _resolve_acceptable_options(row)
        actual_list = row["actual_topics"]
        hint = _primary_hint(row)
        predicted = _primary_label(actual_list)

        topic_predicted[predicted] += 1
        confusion[(hint, predicted)] += 1

        ok = is_acceptable(actual_list, acceptable_options)
        if ok:
            strict_pass += 1
            lenient_pass += 1

        if category not in by_category:
            by_category[category] = {
                "total": 0,
                "pass": 0,
                "failures": [],
            }
        bucket = by_category[category]
        bucket["total"] += 1
        if ok:
            bucket["pass"] += 1
        else:
            bucket["failures"].append(
                {
                    "chunk_id": row["chunk_id"],
                    "acceptable_topics": acceptable_options,
                    "actual": actual_list,
                    "preview": row.get("text_preview", "")[:120],
                }
            )

    total = len(samples)
    recall_by_hint: dict[str, float] = {}
    for hint in sorted({h for h, _ in confusion}):
        total_gold = sum(count for (h, _), count in confusion.items() if h == hint)
        correct = sum(
            1
            for row in samples
            if _primary_hint(row) == hint
            and is_acceptable(row["actual_topics"], _resolve_acceptable_options(row))
        )
        recall_by_hint[hint] = round(correct / total_gold, 4) if total_gold else 0.0

    war_as_predicted = topic_predicted.get("war", 0)

    return {
        "source_report": str(report_path),
        "topics_version": payload.get("topics_version") or payload.get("ontology_version"),
        "total_samples": total,
        "pass_count": strict_pass,
        "pass_rate": round(strict_pass / total, 4) if total else 0.0,
        "war_as_primary": war_as_predicted,
        "war_rate": round(war_as_predicted / total, 4) if total else 0.0,
        "family_as_primary": topic_predicted.get("family", 0),
        "livelihood_as_primary": topic_predicted.get("livelihood", 0),
        "recall_by_category_hint": recall_by_hint,
        "confusion_matrix_hint_to_predicted": [
            {"hint": h, "predicted": p, "count": c}
            for (h, p), c in sorted(confusion.items(), key=lambda item: (-item[1], item[0][0]))
        ],
        "by_category": by_category,
    }


def format_confusion_table(analysis: dict) -> str:
    lines = [
        "Confusion Matrix (category_hint → predicted primary)",
        f"{'Hint':<14} {'Predicted':<14} {'Count':>5}",
        "-" * 36,
    ]
    for row in analysis["confusion_matrix_hint_to_predicted"]:
        lines.append(f"{row['hint']:<14} {row['predicted']:<14} {row['count']:>5}")
    return "\n".join(lines)
