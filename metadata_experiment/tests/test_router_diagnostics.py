from __future__ import annotations

from pathlib import Path

from router_diagnostics import (
    analyze_router_diagnostics,
    first_gold_rank,
    load_book_gold_chunks,
    mrr_at_k,
)


def test_book_gold_manifest_has_ten_questions():
    path = Path(__file__).resolve().parents[1] / "data" / "book_gold_chunks.json"
    questions = load_book_gold_chunks(path)
    assert len(questions) == 10
    assert "Q02" in questions
    assert "c0090" in questions["Q02"]["gold_chunk_ids"]


def test_first_gold_rank_uses_minimum_rank_across_gold_ids():
    retrieved = ["c0006", "c0053", "c0137", "c0136"]
    gold_ids = ["c0053", "c0006"]
    assert first_gold_rank(retrieved, gold_ids) == 1
    assert mrr_at_k(retrieved, gold_ids, k=4) == 1.0


def test_first_gold_rank_returns_none_when_no_gold_present():
    assert first_gold_rank(["c0001", "c0002"], ["c0090"]) is None
    assert mrr_at_k(["c0001", "c0002"], ["c0090"], k=4) == 0.0


def test_first_gold_rank_not_confused_by_gold_list_order():
    retrieved = ["c0101", "c0117", "c0100", "c0116"]
    gold_ids = ["c0117", "c0118", "c0101"]
    assert first_gold_rank(retrieved, gold_ids) == 1
    assert mrr_at_k(retrieved, gold_ids, k=4) == 1.0


def test_analyze_router_diagnostics_summary_keys():
    df, summary = analyze_router_diagnostics()
    assert len(df) == 10
    assert "gold_retention_rate" in summary
    assert "b_miss_gold_filtered_count" in summary
    assert "a_mrr_at_4" in summary
    assert set(df.columns) >= {
        "Question ID",
        "Gold Retained After Filter",
        "A Hit@4",
        "B Hit@4",
        "A MRR@4",
        "B MRR@4",
    }
