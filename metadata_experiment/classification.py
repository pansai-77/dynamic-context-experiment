from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from .classification_prompts import (
    CLASSIFICATION_PROMPT_VERSION,
    build_chunk_classification_prompt_for_text,
)
from .metadata_quality import collect_content_warnings
from .topics import FALLBACK_TOPIC, TOPIC_BY_NAME, TOPIC_TAXONOMY_VERSION, normalize_topic_name

if TYPE_CHECKING:
    from src.llm_mlx import QwenMLX
    from src.models import Chunk


class TopicParseError(ValueError):
    pass


@dataclass(frozen=True)
class ClassificationCacheEntry:
    chunk_id: str
    topics: list[str]
    raw_response: str
    prompt_version: str
    cache_version: str


@dataclass(frozen=True)
class ChunkClassificationResult:
    chunk_id: str
    raw_response: str
    parsed_topics: list[str]
    validation_warnings: tuple[str, ...]
    final_topics: list[str]
    attempts: int
    prompt: str
    prompt_kind: str = "general"
    cache_hit: bool = False
    cache_version: str | None = None
    structure_errors: tuple[str, ...] = field(default_factory=tuple)

    @property
    def topics(self) -> list[str]:
        return self.final_topics


def compute_cache_version(
    *,
    llm_model: str,
    temperature: float,
    max_new_tokens: int,
) -> str:
    payload = {
        "prompt_version": CLASSIFICATION_PROMPT_VERSION,
        "topic_taxonomy_version": TOPIC_TAXONOMY_VERSION,
        "llm_model": llm_model,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return f"{CLASSIFICATION_PROMPT_VERSION}:{digest[:12]}"


class ClassificationCache:
    def __init__(self, cache_dir: Path, cache_version: str) -> None:
        self.cache_dir = cache_dir
        self.cache_version = cache_version
        self.cache_file = cache_dir / f"classification_cache_{cache_version}.json"
        self._entries: dict[str, dict] = {}
        self._load_or_rotate()

    def _load_or_rotate(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        stale_files = sorted(self.cache_dir.glob("classification_cache_*.json"))
        for stale_file in stale_files:
            if stale_file == self.cache_file:
                continue
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            backup_dir = self.cache_dir / "backups" / timestamp
            backup_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(stale_file), str(backup_dir / stale_file.name))

        if self.cache_file.exists():
            self._entries = json.loads(self.cache_file.read_text(encoding="utf-8"))

    def get(self, chunk_id: str) -> ClassificationCacheEntry | None:
        payload = self._entries.get(chunk_id)
        if not payload or payload.get("cache_version") != self.cache_version:
            return None
        topics = list(payload["topics"])
        if not topics or any(topic not in TOPIC_BY_NAME for topic in topics):
            return None
        return ClassificationCacheEntry(
            chunk_id=chunk_id,
            topics=topics,
            raw_response=str(payload["raw_response"]),
            prompt_version=str(payload["prompt_version"]),
            cache_version=str(payload["cache_version"]),
        )

    def put(self, *, chunk_id: str, topics: list[str], raw_response: str) -> None:
        self._entries[chunk_id] = {
            "topics": topics,
            "raw_response": raw_response,
            "prompt_version": CLASSIFICATION_PROMPT_VERSION,
            "cache_version": self.cache_version,
        }
        self.cache_file.write_text(
            json.dumps(self._entries, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def extract_json_object(text: str) -> dict:
    stripped = text.strip()
    if not stripped:
        raise TopicParseError("empty LLM response")

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", stripped)
        if match is None:
            raise TopicParseError("response is not valid JSON") from None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise TopicParseError("response is not valid JSON") from exc

    if not isinstance(payload, dict):
        raise TopicParseError("JSON root must be an object")
    return payload


def parse_validated_topics(raw_response: str, allowed_names: set[str]) -> list[str]:
    payload = extract_json_object(raw_response)
    topics = payload.get("topics")
    if not isinstance(topics, list) or not topics:
        raise TopicParseError("topics must be a non-empty array")

    normalized: list[str] = []
    seen: set[str] = set()
    for item in topics:
        name = normalize_topic_name(str(item).strip())
        if not name:
            raise TopicParseError("topics must not contain blank values")
        if name not in allowed_names:
            raise TopicParseError(f"illegal topic: {name}")
        if name in seen:
            continue
        seen.add(name)
        normalized.append(name)

    if FALLBACK_TOPIC in normalized:
        if len(normalized) != 1:
            raise TopicParseError("fallback topic must be the only topic")
        return normalized

    if not 1 <= len(normalized) <= 2:
        raise TopicParseError(f"expected 1-2 topics, got {len(normalized)}")
    return normalized


def format_retry_message(error: str) -> str:
    return (
        "上一次回答不是合法 JSON，或 JSON 结构不符合要求。\n"
        "请按指定格式重新输出，不要输出解释。"
    )


def is_retryable_parse_error(error: TopicParseError) -> bool:
    message = str(error)
    if message.startswith("illegal topic:"):
        return False
    return True


class ChunkClassificationError(RuntimeError):
    def __init__(self, chunk_id: str, reason: str, attempts: int, last_response: str = "") -> None:
        self.chunk_id = chunk_id
        self.reason = reason
        self.attempts = attempts
        self.last_response = last_response
        super().__init__(
            f"Failed to classify {chunk_id} after {attempts} attempt(s): {reason}"
        )


class ChunkTopicClassifier:
    def __init__(
        self,
        llm: QwenMLX,
        *,
        max_retries: int = 3,
        cache: ClassificationCache | None = None,
    ) -> None:
        self.llm = llm
        self.max_retries = max_retries
        self.allowed_names = set(TOPIC_BY_NAME)
        self.cache = cache
        self.cache_version = cache.cache_version if cache else None

    def classify(self, chunk: Chunk, *, use_cache: bool = True) -> ChunkClassificationResult:
        prompt = build_chunk_classification_prompt_for_text(chunk.text)
        structure_errors: list[str] = []

        if use_cache and self.cache is not None:
            cached = self.cache.get(chunk.chunk_id)
            if cached is not None:
                warnings = collect_content_warnings(chunk.text, cached.topics)
                return ChunkClassificationResult(
                    chunk_id=chunk.chunk_id,
                    raw_response=cached.raw_response,
                    parsed_topics=list(cached.topics),
                    validation_warnings=warnings,
                    final_topics=list(cached.topics),
                    attempts=0,
                    prompt=prompt,
                    cache_hit=True,
                    cache_version=cached.cache_version,
                )

        last_error = "unknown error"
        last_response = ""

        for attempt in range(1, self.max_retries + 1):
            generation = self.llm.answer(prompt)
            last_response = generation.answer
            try:
                parsed_topics = parse_validated_topics(last_response, self.allowed_names)
                warnings = collect_content_warnings(chunk.text, parsed_topics)
                if self.cache is not None:
                    self.cache.put(
                        chunk_id=chunk.chunk_id,
                        topics=parsed_topics,
                        raw_response=last_response,
                    )
                return ChunkClassificationResult(
                    chunk_id=chunk.chunk_id,
                    raw_response=last_response,
                    parsed_topics=parsed_topics,
                    validation_warnings=warnings,
                    final_topics=list(parsed_topics),
                    attempts=attempt,
                    prompt=prompt,
                    structure_errors=tuple(structure_errors),
                )
            except TopicParseError as exc:
                last_error = str(exc)
                structure_errors.append(last_error)
                if not is_retryable_parse_error(exc):
                    raise ChunkClassificationError(
                        chunk.chunk_id,
                        last_error,
                        attempt,
                        last_response,
                    ) from exc
                prompt = (
                    f"{build_chunk_classification_prompt_for_text(chunk.text)}\n\n"
                    f"{format_retry_message(last_error)}"
                )

        raise ChunkClassificationError(
            chunk.chunk_id,
            last_error,
            self.max_retries,
            last_response,
        )


def build_classification_cache(settings) -> ClassificationCache:
    cache_version = compute_cache_version(
        llm_model=settings.llm_model,
        temperature=settings.temperature,
        max_new_tokens=settings.classification_max_new_tokens,
    )
    cache_dir = settings.experiment_dir / "data" / "classification_cache"
    return ClassificationCache(cache_dir, cache_version)


def create_topic_classifier(settings) -> ChunkTopicClassifier:
    from src.llm_mlx import QwenMLX

    print(f"Loading Qwen model ({settings.llm_model}) for chunk classification...")
    llm = QwenMLX(
        settings.llm_model,
        settings.classification_max_new_tokens,
        settings.temperature,
    )
    cache = build_classification_cache(settings)
    print(f"Classification cache version: {cache.cache_version}")
    print("Warming up Qwen model...")
    llm.warm_up()
    return ChunkTopicClassifier(
        llm,
        max_retries=settings.classification_max_retries,
        cache=cache,
    )
