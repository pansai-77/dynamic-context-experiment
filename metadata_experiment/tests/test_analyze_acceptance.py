from __future__ import annotations

import json
from pathlib import Path

from acceptance_analysis import analyze_acceptance_report, is_acceptable


def test_is_acceptable_supports_multiple_valid_labels():
    assert is_acceptable(["family"], [["medical"], ["family"]])
    assert is_acceptable([], [[], ["family"]])
    assert not is_acceptable(["war"], [["family"]])


def test_confusion_matrix_v3_format():
    report = {
        "ontology_version": "3.0",
        "samples": [
            {
                "chunk_id": "p122-c002",
                "category_hint": "family",
                "acceptable_topics": [["family"]],
                "actual_topics": ["family"],
            },
            {
                "chunk_id": "p130-c001",
                "category_hint": "politics",
                "acceptable_topics": [["politics"]],
                "actual_topics": ["war"],
            },
            {
                "chunk_id": "p104-c001",
                "category_hint": "medical",
                "acceptable_topics": [["medical"], ["family"]],
                "actual_topics": ["family"],
            },
        ],
    }
    path = Path("tmp_acceptance_v3.json")
    path.write_text(json.dumps(report), encoding="utf-8")
    try:
        analysis = analyze_acceptance_report(path)
        assert analysis["pass_count"] == 2
        assert analysis["war_as_primary"] == 1
    finally:
        path.unlink(missing_ok=True)
