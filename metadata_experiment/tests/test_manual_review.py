from __future__ import annotations

import csv
from pathlib import Path

import pytest

from metadata_experiment.classification import (
    ChunkClassificationResult,
    TopicParseError,
)
from metadata_experiment.manual_review import (
    ChunkClassificationFailure,
    ManualReviewError,
    export_classification_failures,
    export_manual_override_template,
    load_manual_topic_overrides,
    merge_topics_with_manual_overrides,
    parse_manual_topics_field,
)


def test_parse_manual_topics_field_accepts_pipe_separated_values():
    topics = parse_manual_topics_field("序言与创作背景|老牛与晚年")
    assert topics == ["序言与创作背景", "老牛与晚年"]


def test_parse_manual_topics_field_rejects_illegal_topic():
    with pytest.raises(TopicParseError, match="illegal topic"):
        parse_manual_topics_field("不存在")


def test_load_manual_topic_overrides(tmp_path: Path):
    path = tmp_path / "manual_topic_overrides.csv"
    path.write_text(
        "chunk_id,topics,notes\n"
        "c0006,序言与创作背景,author framing\n",
        encoding="utf-8",
    )
    overrides = load_manual_topic_overrides(path)
    assert overrides["c0006"].topics == ["序言与创作背景"]
    assert overrides["c0006"].notes == "author framing"


def test_merge_topics_with_manual_overrides(tmp_path: Path):
    results = [
        ChunkClassificationResult(
            chunk_id="c0001",
            raw_response='{"topics": ["序言与创作背景"]}',
            parsed_topics=["序言与创作背景"],
            validation_warnings=(),
            final_topics=["序言与创作背景"],
            attempts=1,
            prompt="prompt",
        )
    ]
    failures = [
        ChunkClassificationFailure(
            chunk_id="c0006",
            reason="illegal topic: 贫穷生存压力",
            last_response='{"topics": ["贫穷生存压力"]}',
            attempts=1,
        )
    ]
    manual_path = tmp_path / "manual.csv"
    manual_path.write_text(
        "chunk_id,topics,notes\n"
        "c0006,序言与创作背景|老牛与晚年,manual fix\n",
        encoding="utf-8",
    )
    overrides = load_manual_topic_overrides(manual_path)
    topics, sources = merge_topics_with_manual_overrides(
        results,
        failures,
        overrides,
        all_chunk_ids=["c0001", "c0006"],
    )
    assert topics["c0001"] == ["序言与创作背景"]
    assert topics["c0006"] == ["序言与创作背景", "老牛与晚年"]
    assert sources["c0006"] == "manual"


def test_merge_requires_override_for_each_failure():
    results = [
        ChunkClassificationResult(
            chunk_id="c0001",
            raw_response='{"topics": ["序言与创作背景"]}',
            parsed_topics=["序言与创作背景"],
            validation_warnings=(),
            final_topics=["序言与创作背景"],
            attempts=1,
            prompt="prompt",
        )
    ]
    failures = [
        ChunkClassificationFailure(
            chunk_id="c0006",
            reason="illegal topic: 贫穷生存压力",
            last_response='{"topics": ["贫穷生存压力"]}',
            attempts=1,
        )
    ]
    with pytest.raises(ManualReviewError, match="no manual override"):
        merge_topics_with_manual_overrides(
            results,
            failures,
            {},
            all_chunk_ids=["c0001", "c0006"],
        )


def test_export_failure_and_template_files(tmp_path: Path):
    failures = [
        ChunkClassificationFailure(
            chunk_id="c0006",
            reason="illegal topic: 贫穷生存压力",
            last_response='{"topics": ["贫穷生存压力"]}',
            attempts=1,
        )
    ]
    failure_path = tmp_path / "classification_failures.csv"
    template_path = tmp_path / "manual_topic_overrides.template.csv"
    export_classification_failures(
        failure_path,
        failures,
        text_by_chunk_id={"c0006": "sample text"},
    )
    export_manual_override_template(template_path, failures)
    failure_rows = list(csv.DictReader(failure_path.open(encoding="utf-8")))
    template_rows = list(csv.DictReader(template_path.open(encoding="utf-8")))
    assert failure_rows[0]["chunk_id"] == "c0006"
    assert template_rows[0]["chunk_id"] == "c0006"
    assert template_rows[0]["topics"] == ""
