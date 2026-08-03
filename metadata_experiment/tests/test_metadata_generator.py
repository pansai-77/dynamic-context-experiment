from __future__ import annotations

from dataclasses import dataclass

from acceptance_analysis import analyze_acceptance_report, is_acceptable
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


TOPICS = [TopicDefinition("family", "家庭事件", "婚嫁亲子")]


def test_metadata_generator_rejects_invalid_topics_without_remapping():
    payload = {
        "characters": ["福贵"],
        "topics": ["youqing_death"],
        "keywords": ["献血", "医院", "有庆"],
    }
    metadata, invalid = normalize_metadata_payload(payload, {"medical"})
    assert metadata is None
    assert invalid == ["youqing_death"]


def test_metadata_generator_accepts_single_topic_and_no_importance():
    payload = {
        "characters": ["有庆"],
        "topics": ["medical"],
        "keywords": ["献血", "医院", "抽血"],
    }
    metadata, invalid = normalize_metadata_payload(payload, {"medical", "family"})
    assert invalid == []
    assert metadata is not None
    assert metadata.topics == ["medical"]
    assert metadata.importance is None
    assert metadata.metadata_status == "ok"


def test_metadata_generator_keeps_up_to_two_topics():
    payload = {
        "characters": ["有庆"],
        "topics": ["medical", "family"],
        "keywords": ["一", "二", "三"],
    }
    metadata, invalid = normalize_metadata_payload(payload, {"medical", "family"})
    assert invalid == []
    assert metadata is not None
    assert metadata.topics == ["medical", "family"]


def test_extract_json_object_from_fenced_response():
    text = '```json\n{"topics":["family"],"characters":[],"keywords":["a","b","c"]}\n```'
    payload = extract_json_object(text)
    assert payload["topics"] == ["family"]


def test_metadata_generator_retries_on_invalid_topic():
    invalid = '{"topics":["bad_topic"],"characters":[],"keywords":["a","b","c"]}'
    valid = '{"topics":["family"],"characters":[],"keywords":["a","b","c"]}'
    llm = FakeLLM([invalid, valid])
    result = generate_chunk_metadata(llm, "chunk text", TOPICS, max_retries=1)
    assert result.success is True
    assert result.retries_used == 1


def test_metadata_generator_counts_single_retry_on_persistent_failure():
    invalid = '{"topics":["bad_topic"],"characters":[],"keywords":["a","b","c"]}'
    llm = FakeLLM([invalid, invalid])
    result = generate_chunk_metadata(llm, "chunk text", TOPICS, max_retries=1)
    assert result.success is False
    assert result.retries_used == 1
    assert result.metadata is not None
    assert result.metadata.metadata_status == "failed"
