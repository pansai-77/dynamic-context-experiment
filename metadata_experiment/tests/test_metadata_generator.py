from __future__ import annotations

from dataclasses import dataclass

from metadata_generator import generate_chunk_metadata
from metadata_parsing import extract_json_object, normalize_metadata_payload
from models import TopicDefinition


@dataclass
class FakeGeneration:
    answer: str


class FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.index = 0

    def answer(self, prompt: str) -> FakeGeneration:
        response = self.responses[self.index]
        self.index += 1
        return FakeGeneration(answer=response)


TOPICS = [TopicDefinition("death_loss", "死亡与失去", "亲人离世")]


def test_metadata_generator_rejects_invalid_topics_without_remapping():
    payload = {
        "characters": ["福贵"],
        "topics": ["youqing_death"],
        "keywords": ["献血"],
        "importance": 4,
    }
    metadata, invalid = normalize_metadata_payload(payload, {"disease_medical"})
    assert metadata is None
    assert invalid == ["youqing_death"]


def test_metadata_generator_accepts_valid_topics():
    payload = {
        "characters": ["有庆"],
        "topics": ["disease_medical", "death_loss"],
        "keywords": ["献血"],
        "importance": 5,
    }
    metadata, invalid = normalize_metadata_payload(payload, {"disease_medical", "death_loss"})
    assert invalid == []
    assert metadata is not None
    assert metadata.topics == ["disease_medical", "death_loss"]
    assert metadata.metadata_status == "ok"


def test_extract_json_object_from_fenced_response():
    text = '```json\n{"topics":["death_loss"],"characters":[],"keywords":[],"importance":3}\n```'
    payload = extract_json_object(text)
    assert payload["topics"] == ["death_loss"]


def test_metadata_generator_rejects_invalid_importance():
    payload = {
        "characters": ["有庆"],
        "topics": ["death_loss"],
        "keywords": ["献血"],
        "importance": "high",
    }
    metadata, invalid = normalize_metadata_payload(payload, {"death_loss"})
    assert metadata is None
    assert invalid == []


def test_metadata_generator_accepts_integer_like_importance():
    payload = {
        "characters": ["有庆"],
        "topics": ["death_loss"],
        "keywords": ["献血"],
        "importance": 4.0,
    }
    metadata, invalid = normalize_metadata_payload(payload, {"death_loss"})
    assert invalid == []
    assert metadata is not None
    assert metadata.importance == 4


def test_metadata_generator_counts_retries_not_final_failure():
    invalid = '{"topics":["death_loss"],"characters":[],"keywords":[],"importance":"high"}'
    valid = '{"topics":["death_loss"],"characters":[],"keywords":[],"importance":3}'
    llm = FakeLLM([invalid, valid])
    result = generate_chunk_metadata(llm, "chunk text", TOPICS, max_retries=1)
    assert result.success is True
    assert result.retries_used == 1


def test_metadata_generator_counts_single_retry_on_persistent_failure():
    invalid = '{"topics":["death_loss"],"characters":[],"keywords":[],"importance":"high"}'
    llm = FakeLLM([invalid, invalid])
    result = generate_chunk_metadata(llm, "chunk text", TOPICS, max_retries=1)
    assert result.success is False
    assert result.retries_used == 1
    assert result.metadata is not None
    assert result.metadata.metadata_status == "failed"
