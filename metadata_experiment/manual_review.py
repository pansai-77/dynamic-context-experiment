from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from .classification import ChunkClassificationResult, TopicParseError, parse_validated_topics
from .topics import TOPIC_BY_NAME, topic_names


MANUAL_OVERRIDE_COLUMNS = ("chunk_id", "topics", "notes")
FAILURE_EXPORT_COLUMNS = ("chunk_id", "reason", "last_raw_response", "text_preview")


@dataclass(frozen=True)
class ChunkClassificationFailure:
    chunk_id: str
    reason: str
    last_response: str
    attempts: int


@dataclass(frozen=True)
class ManualTopicOverride:
    chunk_id: str
    topics: list[str]
    notes: str = ""


class ManualReviewError(ValueError):
    pass


def parse_manual_topics_field(raw: str, allowed_names: set[str] | None = None) -> list[str]:
    names = allowed_names or set(TOPIC_BY_NAME)
    parts = [part.strip() for part in raw.split("|") if part.strip()]
    if not parts:
        raise TopicParseError("manual topics must not be blank")
    synthetic = json.dumps({"topics": parts}, ensure_ascii=False)
    return parse_validated_topics(synthetic, names)


def load_manual_topic_overrides(path: Path) -> dict[str, ManualTopicOverride]:
    if not path.exists():
        return {}

    overrides: dict[str, ManualTopicOverride] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in MANUAL_OVERRIDE_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ManualReviewError(
                f"{path} is missing required columns: {', '.join(missing)}"
            )

        for row_number, row in enumerate(reader, start=2):
            chunk_id = (row.get("chunk_id") or "").strip()
            topics_raw = (row.get("topics") or "").strip()
            notes = (row.get("notes") or "").strip()
            if not chunk_id and not topics_raw and not notes:
                continue
            if not chunk_id:
                raise ManualReviewError(f"{path}:{row_number}: chunk_id is required")
            if not topics_raw:
                raise ManualReviewError(
                    f"{path}:{row_number}: topics is required for {chunk_id}"
                )
            try:
                topics = parse_manual_topics_field(topics_raw)
            except TopicParseError as exc:
                raise ManualReviewError(
                    f"{path}:{row_number}: invalid topics for {chunk_id}: {exc}"
                ) from exc
            if chunk_id in overrides:
                raise ManualReviewError(
                    f"{path}:{row_number}: duplicate manual override for {chunk_id}"
                )
            overrides[chunk_id] = ManualTopicOverride(
                chunk_id=chunk_id,
                topics=topics,
                notes=notes,
            )
    return overrides


def export_classification_failures(
    path: Path,
    failures: list[ChunkClassificationFailure],
    *,
    text_by_chunk_id: dict[str, str],
    preview_chars: int = 160,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FAILURE_EXPORT_COLUMNS))
        writer.writeheader()
        for failure in failures:
            preview = text_by_chunk_id.get(failure.chunk_id, "").replace("\n", " ")
            if len(preview) > preview_chars:
                preview = preview[:preview_chars] + "..."
            writer.writerow({
                "chunk_id": failure.chunk_id,
                "reason": failure.reason,
                "last_raw_response": failure.last_response,
                "text_preview": preview,
            })


def export_manual_override_template(
    path: Path,
    failures: list[ChunkClassificationFailure],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MANUAL_OVERRIDE_COLUMNS))
        writer.writeheader()
        for failure in failures:
            writer.writerow({
                "chunk_id": failure.chunk_id,
                "topics": "",
                "notes": failure.reason,
            })


def merge_topics_with_manual_overrides(
    results: list[ChunkClassificationResult],
    failures: list[ChunkClassificationFailure],
    manual_overrides: dict[str, ManualTopicOverride],
    *,
    all_chunk_ids: list[str],
) -> tuple[dict[str, list[str]], dict[str, str]]:
    topics_by_chunk_id: dict[str, list[str]] = {
        result.chunk_id: list(result.final_topics) for result in results
    }
    source_by_chunk_id: dict[str, str] = {
        result.chunk_id: "llm" for result in results
    }

    failure_ids = {failure.chunk_id for failure in failures}
    for chunk_id, override in manual_overrides.items():
        if chunk_id not in failure_ids:
            raise ManualReviewError(
                f"Manual override for {chunk_id} is not allowed: chunk did not fail LLM classification"
            )
        topics_by_chunk_id[chunk_id] = list(override.topics)
        source_by_chunk_id[chunk_id] = "manual"

    missing = [chunk_id for chunk_id in all_chunk_ids if chunk_id not in topics_by_chunk_id]
    unresolved_failures = [
        failure.chunk_id
        for failure in failures
        if failure.chunk_id not in manual_overrides
    ]
    if unresolved_failures:
        raise ManualReviewError(
            "LLM classification failed and no manual override was provided for: "
            + ", ".join(unresolved_failures)
        )

    if missing:
        raise ManualReviewError(
            "Missing topics for chunks: "
            + ", ".join(missing[:12])
            + (" ..." if len(missing) > 12 else "")
        )

    return topics_by_chunk_id, source_by_chunk_id


def format_manual_review_summary(
    *,
    results: list[ChunkClassificationResult],
    failures: list[ChunkClassificationFailure],
    manual_overrides: dict[str, ManualTopicOverride],
    topics_by_chunk_id: dict[str, list[str]] | None = None,
    source_by_chunk_id: dict[str, str] | None = None,
) -> str:
    lines = [
        "Manual review summary",
        f"  LLM success: {len(results)}",
        f"  LLM failure: {len(failures)}",
        f"  Manual overrides loaded: {len(manual_overrides)}",
    ]
    if topics_by_chunk_id is not None and source_by_chunk_id is not None:
        manual_count = sum(1 for source in source_by_chunk_id.values() if source == "manual")
        lines.append(f"  Final manual topics applied: {manual_count}")
        lines.append(f"  Final chunks covered: {len(topics_by_chunk_id)}")
    lines.append("")
    lines.append("Allowed topics:")
    lines.extend(f"  - {name}" for name in topic_names())
    return "\n".join(lines)
