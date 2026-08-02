from __future__ import annotations

from metadata_parsing import extract_json_object, normalize_metadata_payload


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
