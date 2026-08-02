from __future__ import annotations

import json
import re

from models import ChunkMetadata


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


def parse_importance(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("importance must be an integer, not a boolean.")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise ValueError("importance must be an integer.")
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError("importance must be an integer.")
        parsed = float(stripped)
        if parsed.is_integer():
            return int(parsed)
        raise ValueError("importance must be an integer.")
    raise ValueError("importance must be an integer.")


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
    keywords = payload.get("keywords") or []
    if not isinstance(keywords, list):
        keywords = []

    try:
        importance = parse_importance(payload.get("importance"))
    except (ValueError, TypeError):
        return None, []

    metadata = ChunkMetadata(
        characters=[str(item).strip() for item in characters if str(item).strip()],
        topics=topic_ids[:2],
        keywords=[str(item).strip() for item in keywords if str(item).strip()],
        importance=importance,
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
