from __future__ import annotations

import json
import re

from models import ChunkMetadata

KEYWORD_COUNT = 3
MAX_TOPICS = 2


def normalize_topic_ids(topic_ids: list[str]) -> list[str]:
    """Preserve LLM topic order; dedupe and cap count only."""
    normalized: list[str] = []
    for topic_id in topic_ids:
        if topic_id not in normalized:
            normalized.append(topic_id)
        if len(normalized) >= MAX_TOPICS:
            break
    return normalized


def extract_json_object(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model response.")
    return json.loads(stripped[start : end + 1])


def normalize_keywords(keywords: list) -> list[str]:
    if not isinstance(keywords, list):
        return []
    cleaned = [str(item).strip() for item in keywords if str(item).strip()]
    return cleaned[:KEYWORD_COUNT]


def normalize_metadata_payload(
    payload: dict,
    allowed_topic_ids: set[str],
) -> tuple[ChunkMetadata | None, list[str]]:
    topics = payload.get("topics") or []
    if not isinstance(topics, list):
        topics = []
    topic_ids = [str(item).strip() for item in topics if str(item).strip()]
    invalid_topic_ids = sorted({topic_id for topic_id in topic_ids if topic_id not in allowed_topic_ids})
    if invalid_topic_ids:
        return None, invalid_topic_ids

    characters = payload.get("characters") or []
    if not isinstance(characters, list):
        characters = []
    keywords = normalize_keywords(payload.get("keywords") or [])

    metadata = ChunkMetadata(
        characters=[str(item).strip() for item in characters if str(item).strip()][:4],
        topics=normalize_topic_ids(topic_ids),
        keywords=keywords,
        importance=None,
        metadata_status="ok",
    )
    return metadata, []


def failed_metadata() -> ChunkMetadata:
    return ChunkMetadata(
        characters=[],
        topics=[],
        keywords=[],
        importance=None,
        metadata_status="failed",
    )
