from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .topics import BROAD_TOPICS, FALLBACK_TOPIC, PREFACE_TOPIC, SPECIFIC_TOPICS, topic_names


DIAGNOSTIC_CHUNK_IDS = (
    "c0004",
    "c0007",
    "c0015",
    "c0037",
    "c0062",
    "c0090",
    "c0122",
    "c0124",
    "c0128",
    "c0135",
    "c0136",
    "c0137",
)

DIAGNOSTIC_CHUNK_EXPECTATIONS: dict[str, dict[str, object]] = {
    "c0004": {"must_equal": ["序言与创作背景"]},
    "c0007": {"must_not_include": ["贫困生计", "赌博败家"], "may_include": ["家庭生活"]},
    "c0015": {"must_include": ["赌博败家"], "may_include": ["家庭生活"]},
    "c0037": {"must_include": ["参军战争"]},
    "c0062": {"must_include": ["人民公社"]},
    "c0090": {"must_include": ["有庆经历", "医疗献血"]},
    "c0122": {"must_include": ["凤霞经历", "医疗献血"]},
    "c0124": {"must_include": ["家珍病逝"]},
    "c0128": {"must_include": ["二喜苦根", "死亡苦难"]},
    "c0135": {"must_include": ["二喜苦根", "死亡苦难"]},
    "c0136": {"must_include": ["老牛陪伴"]},
    "c0137": {"must_include": ["老牛陪伴"]},
}

BROAD_TOPIC_COVERAGE_WARN_RATIO = 0.50

COPYRIGHT_MARKERS = (
    "排版软件",
    "二十四小时内删除",
    "本文件制作者",
    "软件技术交流",
    "购买正版书籍",
    "删除发布文件",
)

PREFACE_MARKERS = (
    "一位真正的作家",
    "福克纳",
    "文学现实",
    "《老黑奴》",
    "我决定写下一篇",
    "写作过程让我明白",
    "前言",
)

MEDICAL_CUES = ("医院", "医生", "大夫", "护士", "抽血", "献血", "验到", "血型")
GAMBLING_CUES = ("赌坊", "赌钱", "赌场", "赌博", "赌")


@dataclass(frozen=True)
class DistributionQualityReport:
    warnings: tuple[str, ...]
    topic_counts: Counter[str]
    topic_to_chunk_ids: dict[str, list[str]]


@dataclass(frozen=True)
class QualitySampleRecord:
    chunk_id: str
    topics: list[str]
    validation_warnings: list[str]
    manual_check_notes: str = ""


def _contains_any(text: str, cues: tuple[str, ...]) -> bool:
    return any(cue in text for cue in cues)


def audit_content_warnings(chunk_text: str, topics: list[str]) -> list[str]:
    """Development/audit warnings only. Must not modify parsed topics."""
    warnings: list[str] = []

    if any(marker in chunk_text for marker in COPYRIGHT_MARKERS) and topics != [FALLBACK_TOPIC]:
        warnings.append("text appears to contain copyright/footer markers but topics are not fallback-only")

    if any(marker in chunk_text for marker in PREFACE_MARKERS) and "福贵" not in chunk_text:
        if PREFACE_TOPIC not in topics:
            warnings.append("text appears to be preface/author background without preface topic")

    if "医疗献血" in topics and not _contains_any(chunk_text, MEDICAL_CUES):
        warnings.append("topic 医疗献血 assigned without clear medical/blood-donation vocabulary")

    if "赌博败家" in topics and not _contains_any(chunk_text, GAMBLING_CUES):
        warnings.append("topic 赌博败家 assigned without clear gambling vocabulary")

    if _contains_any(chunk_text, ("壮丁", "拉大炮", "国民党大兵", "去拉大炮")):
        if "参军战争" not in topics and FALLBACK_TOPIC not in topics and PREFACE_TOPIC not in topics:
            warnings.append("text mentions conscription/wartime service but 参军战争 is absent")

    if _contains_any(chunk_text, ("吃食堂", "办了食堂", "村里办起了食堂", "炼钢", "砸锅")):
        if "人民公社" not in topics and FALLBACK_TOPIC not in topics and PREFACE_TOPIC not in topics:
            warnings.append("text mentions commune/canteen activity but 人民公社 is absent")

    broad_hits = [topic for topic in topics if topic in BROAD_TOPICS]
    specific_hits = [topic for topic in topics if topic in SPECIFIC_TOPICS]
    if broad_hits and not specific_hits and _contains_any(
        chunk_text,
        ("壮丁", "抽血", "献血", "食堂", "砸锅", "水泥板", "买牛", "赌坊"),
    ):
        warnings.append("only broad topics assigned despite specific-event cues in text")

    return warnings


def build_topic_to_chunk_ids(topics_by_chunk_id: dict[str, list[str]]) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = defaultdict(list)
    for chunk_id, topics in topics_by_chunk_id.items():
        for topic in topics:
            mapping[topic].append(chunk_id)
    for topic in mapping:
        mapping[topic].sort()
    return dict(mapping)


def evaluate_distribution_quality(
    topics_by_chunk_id: dict[str, list[str]],
    *,
    chunk_count: int | None = None,
) -> DistributionQualityReport:
    total_chunks = chunk_count or len(topics_by_chunk_id)
    topic_counts = Counter()
    for topics in topics_by_chunk_id.values():
        topic_counts.update(topics)
    topic_to_chunk_ids = build_topic_to_chunk_ids(topics_by_chunk_id)

    warnings: list[str] = []

    zero_count_topics = [topic for topic in topic_names() if topic_counts.get(topic, 0) == 0]
    if zero_count_topics:
        warnings.append(f"以下允许主题计数为 0：{', '.join(zero_count_topics)}")

    for topic in BROAD_TOPICS:
        covered_chunks = len(topic_to_chunk_ids.get(topic, []))
        if total_chunks and covered_chunks / total_chunks > BROAD_TOPIC_COVERAGE_WARN_RATIO:
            warnings.append(
                f"宽泛主题“{topic}”覆盖 {covered_chunks}/{total_chunks} "
                f"({covered_chunks / total_chunks:.1%})，超过 {BROAD_TOPIC_COVERAGE_WARN_RATIO:.0%} 阈值"
            )

    suspicious_medical = len(topic_to_chunk_ids.get("医疗献血", []))
    if total_chunks and suspicious_medical / total_chunks > 0.20:
        warnings.append(
            f"“医疗献血”覆盖 {suspicious_medical}/{total_chunks} "
            f"({suspicious_medical / total_chunks:.1%})，可能存在异常泛滥"
        )

    return DistributionQualityReport(
        warnings=tuple(warnings),
        topic_counts=topic_counts,
        topic_to_chunk_ids=topic_to_chunk_ids,
    )


def format_distribution_report(report: DistributionQualityReport) -> str:
    lines = ["Topic distribution with chunk IDs:"]
    for topic in topic_names():
        chunk_ids = report.topic_to_chunk_ids.get(topic, [])
        lines.append(f"  {topic}: {len(chunk_ids)} -> {', '.join(chunk_ids) if chunk_ids else '-'}")
    if report.warnings:
        lines.append("Warnings:")
        lines.extend(f"  - {item}" for item in report.warnings)
    return "\n".join(lines)


def evaluate_diagnostic_expectations(chunk_id: str, topics: list[str]) -> list[str]:
    """Development-only expectations for manual inspection. Not a build gate."""
    expectation = DIAGNOSTIC_CHUNK_EXPECTATIONS.get(chunk_id)
    if expectation is None:
        return []

    failures: list[str] = []
    expected_equal = expectation.get("must_equal")
    if expected_equal is not None and topics != list(expected_equal):
        failures.append(f"期望 topics={list(expected_equal)}，实际={topics}")

    for topic in expectation.get("must_include", ()):
        if topic not in topics:
            failures.append(f"缺少必须主题“{topic}”")

    for topic in expectation.get("must_not_include", ()):
        if topic in topics:
            failures.append(f"不应包含“{topic}”")

    return failures


def load_or_create_quality_sample(
    *,
    all_chunk_ids: list[str],
    sample_file: Path,
    sample_size: int,
    seed: int,
) -> list[str]:
    if sample_file.exists():
        payload = json.loads(sample_file.read_text(encoding="utf-8"))
        return list(payload["chunk_ids"])

    excluded = set(DIAGNOSTIC_CHUNK_IDS)
    candidates = sorted(chunk_id for chunk_id in all_chunk_ids if chunk_id not in excluded)
    rng = random.Random(seed)
    chosen = rng.sample(candidates, k=min(sample_size, len(candidates)))
    chosen.sort()
    sample_file.parent.mkdir(parents=True, exist_ok=True)
    sample_file.write_text(
        json.dumps(
            {
                "seed": seed,
                "sample_size": sample_size,
                "excluded_development_chunks": sorted(excluded),
                "chunk_ids": chosen,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return chosen


def format_quality_sample_report(records: list[QualitySampleRecord]) -> str:
    lines = ["Independent quality sample check:"]
    for record in records:
        lines.append(f"- {record.chunk_id}: topics={record.topics}")
        if record.validation_warnings:
            lines.append(f"  warnings: {record.validation_warnings}")
        if record.manual_check_notes:
            lines.append(f"  manual_notes: {record.manual_check_notes}")
        else:
            lines.append("  manual_notes: <pending human review>")
    return "\n".join(lines)
