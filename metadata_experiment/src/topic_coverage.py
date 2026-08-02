from __future__ import annotations

from collections import Counter

from models import ChunkMetadata

CORE_TOPICS_FOR_REVIEW = (
    "death_loss",
    "suffering_survival",
    "parent_child",
    "marriage_family",
)

WARNING_SINGLE_TOPIC_THRESHOLD = 0.70
WARNING_AVG_TOPICS_THRESHOLD = 1.80
WARNING_EMPTY_TOPIC_RATIO_THRESHOLD = 0.35


def build_topic_coverage_report(
    metadata_by_chunk_id: dict[str, ChunkMetadata],
    allowed_topic_ids: set[str],
) -> dict:
    total_chunks = len(metadata_by_chunk_id)
    metadata_ok = [m for m in metadata_by_chunk_id.values() if m.metadata_status == "ok"]
    metadata_failed = [m for m in metadata_by_chunk_id.values() if m.metadata_status == "failed"]
    empty_topic_ok = [m for m in metadata_ok if not m.topics]

    topic_counter: Counter[str] = Counter()
    topic_total = 0
    for metadata in metadata_ok:
        for topic_id in metadata.topics:
            topic_counter[topic_id] += 1
            topic_total += 1

    topic_counts = {topic_id: topic_counter.get(topic_id, 0) for topic_id in sorted(allowed_topic_ids)}
    topic_coverage = {
        topic_id: round(topic_counts[topic_id] / total_chunks, 4) if total_chunks else 0.0
        for topic_id in sorted(allowed_topic_ids)
    }

    return {
        "total_chunks": total_chunks,
        "metadata_ok_count": len(metadata_ok),
        "metadata_failed_count": len(metadata_failed),
        "empty_topic_ok_count": len(empty_topic_ok),
        "avg_topics_per_chunk": round(topic_total / total_chunks, 4) if total_chunks else 0.0,
        "topic_counts": topic_counts,
        "topic_coverage": topic_coverage,
    }


def topic_coverage_warnings(report: dict) -> list[str]:
    warnings: list[str] = []
    total = report["total_chunks"]
    if total == 0:
        return warnings

    for topic_id, coverage in report["topic_coverage"].items():
        if coverage > WARNING_SINGLE_TOPIC_THRESHOLD:
            warnings.append(
                f"WARNING: {topic_id} covers {coverage:.1%} of chunks; inspect for possible over-labeling."
            )

    for topic_id in CORE_TOPICS_FOR_REVIEW:
        count = report["topic_counts"].get(topic_id, 0)
        if count == 0:
            warnings.append(
                f"WARNING: {topic_id} covers 0 chunks; inspect prompt/topic consistency."
            )

    if report["avg_topics_per_chunk"] >= WARNING_AVG_TOPICS_THRESHOLD:
        warnings.append(
            f"WARNING: avg_topics_per_chunk={report['avg_topics_per_chunk']:.2f}; "
            "inspect for possible over-labeling."
        )

    empty_ratio = report["empty_topic_ok_count"] / total
    if empty_ratio >= WARNING_EMPTY_TOPIC_RATIO_THRESHOLD:
        warnings.append(
            f"WARNING: {empty_ratio:.1%} of chunks have topics=[] with metadata_status=ok; "
            "inspect whether topic definitions are too narrow."
        )

    if report["metadata_failed_count"] / total >= 0.05:
        warnings.append(
            f"WARNING: metadata_failed_count={report['metadata_failed_count']}; "
            "inspect JSON parsing and field validation."
        )

    return warnings
