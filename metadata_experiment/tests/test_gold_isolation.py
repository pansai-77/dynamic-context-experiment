from __future__ import annotations

from pathlib import Path


def test_runner_sources_do_not_reference_gold_files():
    experiment_root = Path(__file__).resolve().parents[1]
    checked = [
        experiment_root / "scripts" / "run_metadata_experiment.py",
        experiment_root / "src" / "experiment.py",
    ]
    forbidden = ("gold_chunks", "gold_topics", "Hit@4", "MRR@4")
    for path in checked:
        source = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in source, f"{token} found in {path.name}"
