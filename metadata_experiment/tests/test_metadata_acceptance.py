from __future__ import annotations

import json

from config import settings


def test_acceptance_manifest_has_40_unique_chunks():
    manifest = json.loads(settings.metadata_acceptance_samples_file.read_text(encoding="utf-8"))
    chunk_ids: list[str] = []
    for category in manifest["categories"]:
        assert len(category["chunk_ids"]) == 5
        chunk_ids.extend(category["chunk_ids"])
    assert len(chunk_ids) == 40
    assert len(set(chunk_ids)) == 40


def test_allowed_topics_v23_all_topics_have_boundaries():
    payload = json.loads(settings.allowed_topics_file.read_text(encoding="utf-8"))
    assert payload["version"] == "2.3"
    for item in payload["topics"]:
        description = item["description"]
        assert "使用条件" in description, item["id"]
        assert "不得使用" in description, item["id"]
        assert "边界" in description, item["id"]
