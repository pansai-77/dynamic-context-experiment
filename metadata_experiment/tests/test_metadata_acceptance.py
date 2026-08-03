from __future__ import annotations

import json

from config import settings


def test_acceptance_manifest_v3_has_40_unique_chunks():
    manifest = json.loads(settings.metadata_acceptance_samples_file.read_text(encoding="utf-8"))
    assert manifest["version"] == "3.0"
    chunk_ids = [sample["chunk_id"] for sample in manifest["samples"]]
    assert len(chunk_ids) == 40
    assert len(set(chunk_ids)) == 40
    for sample in manifest["samples"]:
        assert "acceptable_topics" in sample
        assert "category_hint" in sample


def test_allowed_topics_v32_has_seven_event_topics():
    payload = json.loads(settings.allowed_topics_file.read_text(encoding="utf-8"))
    assert payload["version"] == "3.2"
    ids = [item["id"] for item in payload["topics"]]
    assert set(ids) == {
        "war",
        "politics",
        "gambling",
        "family",
        "medical",
        "labor",
        "livelihood",
    }
    by_id = {item["id"]: item for item in payload["topics"]}
    assert by_id["family"]["label"] == "家庭关系与家庭事件"
    assert "土地改革" in by_id["politics"]["description"]
    assert "验血" in by_id["medical"]["description"]
