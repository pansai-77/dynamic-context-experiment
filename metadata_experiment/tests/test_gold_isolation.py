from __future__ import annotations

from pathlib import Path


def test_run_script_does_not_load_gold_manifest():
    experiment_root = Path(__file__).resolve().parents[1]
    source = (experiment_root / "scripts" / "run_metadata_experiment.py").read_text(encoding="utf-8")
    forbidden = ("book_gold_chunks", "gold_topics", "router_diagnostics")
    for token in forbidden:
        assert token not in source, f"{token} found in run_metadata_experiment.py"
