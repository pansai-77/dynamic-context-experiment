from __future__ import annotations

import json
from dataclasses import dataclass

from metadata_parsing import extract_json_object, failed_metadata, normalize_metadata_payload
from models import ChunkMetadata, TopicDefinition
from prompts import build_metadata_prompt


@dataclass(frozen=True)
class MetadataGenerationResult:
    metadata: ChunkMetadata | None
    success: bool
    json_parse_failure: bool
    invalid_topic_ids: list[str]
    retries_used: int
    raw_response: str = ""


def generate_chunk_metadata(
    llm,
    chunk_text: str,
    topics: list[TopicDefinition],
    max_retries: int = 1,
) -> MetadataGenerationResult:
    allowed_topic_ids = {topic.id for topic in topics}
    retries_used = 0
    last_raw = ""
    saw_json_failure = False
    invalid_topic_ids: list[str] = []

    attempts = max_retries + 1
    for attempt_index in range(attempts):
        prompt = build_metadata_prompt(chunk_text, topics, retry=attempt_index > 0)
        generation = llm.answer(prompt)
        last_raw = generation.answer
        try:
            payload = extract_json_object(generation.answer)
        except (ValueError, json.JSONDecodeError):
            saw_json_failure = True
            if attempt_index < attempts - 1:
                retries_used += 1
            continue

        metadata, invalid_topic_ids = normalize_metadata_payload(payload, allowed_topic_ids)
        if metadata is not None:
            return MetadataGenerationResult(
                metadata=metadata,
                success=True,
                json_parse_failure=saw_json_failure,
                invalid_topic_ids=[],
                retries_used=retries_used,
                raw_response=last_raw,
            )
        if attempt_index < attempts - 1:
            retries_used += 1

    return MetadataGenerationResult(
        metadata=failed_metadata(),
        success=False,
        json_parse_failure=saw_json_failure,
        invalid_topic_ids=invalid_topic_ids,
        retries_used=retries_used,
        raw_response=last_raw,
    )
