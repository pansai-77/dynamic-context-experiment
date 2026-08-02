from __future__ import annotations

from models import ChunkMetadata
from topic_coverage import build_topic_coverage_report, topic_coverage_warnings


def _metadata(chunk_id: str, topics: list[str], status: str = "ok") -> ChunkMetadata:
    return ChunkMetadata(
        characters=[],
        topics=topics,
        keywords=[],
        importance=3 if status == "ok" else None,
        metadata_status=status,
    )


def test_empty_topics_with_ok_status():
    payload = {
        "characters": [],
        "topics": [],
        "keywords": ["日常"],
        "importance": 2,
    }
    from metadata_parsing import normalize_metadata_payload

    metadata, invalid = normalize_metadata_payload(payload, {"death_loss", "parent_child"})
    assert invalid == []
    assert metadata is not None
    assert metadata.topics == []
    assert metadata.metadata_status == "ok"


def test_topic_coverage_report_counts_empty_ok_and_failed():
    allowed = {"death_loss", "parent_child", "marriage_family"}
    metadata_by_chunk_id = {
        "c1": _metadata("c1", ["death_loss"]),
        "c2": _metadata("c2", []),
        "c3": _metadata("c3", ["parent_child", "death_loss"]),
        "c4": _metadata("c4", [], status="failed"),
    }
    report = build_topic_coverage_report(metadata_by_chunk_id, allowed)
    assert report["total_chunks"] == 4
    assert report["metadata_ok_count"] == 3
    assert report["metadata_failed_count"] == 1
    assert report["empty_topic_ok_count"] == 1
    assert report["avg_topics_per_chunk"] == 0.75
    assert report["topic_counts"]["death_loss"] == 2
    assert report["topic_counts"]["parent_child"] == 1
    assert report["topic_coverage"]["death_loss"] == 0.5


def test_topic_coverage_warnings_detect_overlabel_and_zero_coverage():
    report = {
        "total_chunks": 100,
        "metadata_ok_count": 98,
        "metadata_failed_count": 2,
        "empty_topic_ok_count": 5,
        "avg_topics_per_chunk": 0.9,
        "topic_counts": {
            "death_loss": 85,
            "parent_child": 0,
            "marriage_family": 10,
            "suffering_survival": 12,
        },
        "topic_coverage": {
            "death_loss": 0.85,
            "parent_child": 0.0,
            "marriage_family": 0.10,
            "suffering_survival": 0.12,
        },
    }
    warnings = topic_coverage_warnings(report)
    assert any("death_loss" in warning and "over-labeling" in warning for warning in warnings)
    assert any("parent_child" in warning and "0 chunks" in warning for warning in warnings)
