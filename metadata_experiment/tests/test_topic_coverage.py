from __future__ import annotations

from models import ChunkMetadata
from topic_coverage import build_topic_coverage_report, topic_coverage_warnings


def _metadata(chunk_id: str, topics: list[str], status: str = "ok") -> ChunkMetadata:
    return ChunkMetadata(
        characters=[],
        topics=topics,
        keywords=["a", "b", "c"],
        importance=None,
        metadata_status=status,
    )


def test_empty_topics_with_ok_status():
    payload = {
        "characters": [],
        "topics": [],
        "keywords": ["日常", "对话", "过渡"],
    }
    from metadata_parsing import normalize_metadata_payload

    metadata, invalid = normalize_metadata_payload(payload, {"family", "war"})
    assert invalid == []
    assert metadata is not None
    assert metadata.topics == []
    assert metadata.metadata_status == "ok"


def test_topic_coverage_report_counts_empty_ok_and_failed():
    allowed = {"war", "family", "medical"}
    metadata_by_chunk_id = {
        "c1": _metadata("c1", ["war"]),
        "c2": _metadata("c2", []),
        "c3": _metadata("c3", ["family"]),
        "c4": _metadata("c4", [], status="failed"),
    }
    report = build_topic_coverage_report(metadata_by_chunk_id, allowed)
    assert report["total_chunks"] == 4
    assert report["metadata_ok_count"] == 3
    assert report["metadata_failed_count"] == 1
    assert report["empty_topic_ok_count"] == 1
    assert report["avg_topics_per_chunk"] == 0.5
    assert report["topic_counts"]["war"] == 1
    assert report["topic_counts"]["family"] == 1


def test_topic_coverage_warnings_detect_overlabel_and_zero_coverage():
    report = {
        "total_chunks": 100,
        "metadata_ok_count": 98,
        "metadata_failed_count": 2,
        "empty_topic_ok_count": 5,
        "avg_topics_per_chunk": 0.9,
        "topic_counts": {
            "war": 85,
            "family": 0,
            "politics": 10,
            "medical": 12,
            "labor": 8,
            "livelihood": 6,
            "gambling": 5,
        },
        "topic_coverage": {
            "war": 0.85,
            "family": 0.0,
            "politics": 0.10,
            "medical": 0.12,
            "labor": 0.08,
            "livelihood": 0.06,
            "gambling": 0.05,
        },
    }
    warnings = topic_coverage_warnings(report)
    assert any("war" in warning and "over-labeling" in warning for warning in warnings)
    assert any("family" in warning and "0 chunks" in warning for warning in warnings)
