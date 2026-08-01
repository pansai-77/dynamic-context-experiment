from __future__ import annotations

import pandas as pd
import pytest

from src.results_io import (
    filter_questions,
    format_detailed_dataframe,
    parse_question_ids,
    patch_detailed_results,
)

def _sample_questions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Question ID": ["Q01", "Q02", "Q16", "Q17"],
            "Question Type": ["Book", "Book", "Rewrite", "Rewrite"],
            "Question": ["q1", "q2", "q16", "q17"],
            "Ground Truth": ["gt1", "gt2", "gt16", "gt17"],
            "Source": ["正文", "正文", "Rewrite", "Rewrite"],
        }
    )

def _sample_result_row(
    question_id: str,
    question_type: str,
    method: str,
    answer: str,
) -> dict:
    return {
        "question_id": question_id,
        "question_type": question_type,
        "question": question_id.lower(),
        "method": method,
        "top_k": 4,
        "used_retrieval": True,
        "input_tokens": 100,
        "output_tokens": 20,
        "total_tokens": 120,
        "retrieval_time_ms": 10.0,
        "llm_time_ms": 100.0,
        "total_time_ms": 110.0,
        "tokens_per_second": 200.0,
        "estimated_cost_usd": 0.00001,
        "answer": answer,
        "retrieved_chunks": 4,
        "retrieved_sources": "p1:0.900",
    }

def test_parse_question_ids_splits_and_trims():
    assert parse_question_ids("Q16, Q17,Q18") == ["Q16", "Q17", "Q18"]
    assert parse_question_ids("") is None
    assert parse_question_ids(None) is None

def test_filter_questions_preserves_requested_order():
    filtered = filter_questions(_sample_questions(), ["Q17", "Q16"])
    assert filtered["Question ID"].tolist() == ["Q17", "Q16"]

def test_filter_questions_rejects_unknown_ids():
    with pytest.raises(ValueError, match="Unknown question ID"):
        filter_questions(_sample_questions(), ["Q99"])

def test_patch_detailed_results_replaces_rows_and_clears_scores(tmp_path):
    existing = pd.DataFrame(
        [
            {
                "Question ID": "Q16",
                "Question Type": "Rewrite",
                "Question": "old question",
                "Method": "Baseline (Top-8)",
                "Top-k": 8,
                "Used Retrieval": True,
                "Input Tokens": 1000,
                "Output Tokens": 50,
                "Total Tokens": 1050,
                "Retrieval Time(ms)": 100.0,
                "LLM Time(ms)": 2000.0,
                "Total Time(ms)": 2100.0,
                "Output Tokens/sec": 25.0,
                "Estimated Cost(USD)": 0.0001,
                "Answer": "old answer",
                "Retrieved Chunks": 8,
                "Retrieved Sources": "p1:0.900",
                "Score(0-3)": 3,
                "Notes": "keep me cleared on replaced row",
            },
            {
                "Question ID": "Q01",
                "Question Type": "Book",
                "Question": "book question",
                "Method": "Baseline (Top-8)",
                "Top-k": 8,
                "Used Retrieval": True,
                "Input Tokens": 900,
                "Output Tokens": 40,
                "Total Tokens": 940,
                "Retrieval Time(ms)": 90.0,
                "LLM Time(ms)": 1800.0,
                "Total Time(ms)": 1890.0,
                "Output Tokens/sec": 22.0,
                "Estimated Cost(USD)": 0.00009,
                "Answer": "unchanged answer",
                "Retrieved Chunks": 8,
                "Retrieved Sources": "p2:0.800",
                "Score(0-3)": 2,
                "Notes": "should stay",
            },
        ]
    )
    detailed_path = tmp_path / "detailed_results.xlsx"
    existing.to_excel(detailed_path, index=False, sheet_name="Detailed Results")

    incoming = pd.DataFrame(
        [
            _sample_result_row("Q16", "Rewrite", "Baseline (Top-8)", "new answer"),
        ]
    )
    patched = patch_detailed_results(detailed_path, incoming)

    assert len(patched) == 2
    q16 = patched[patched["Question ID"] == "Q16"].iloc[0]
    q01 = patched[patched["Question ID"] == "Q01"].iloc[0]
    assert q16["Answer"] == "new answer"
    assert pd.isna(q16["Score(0-3)"])
    assert q16["Notes"] == ""
    assert q01["Answer"] == "unchanged answer"
    assert q01["Score(0-3)"] == 2
    assert q01["Notes"] == "should stay"

def test_format_detailed_dataframe_adds_score_and_notes_columns():
    formatted = format_detailed_dataframe(
        pd.DataFrame([_sample_result_row("Q16", "Rewrite", "No RAG", "answer")])
    )
    assert "Score(0-3)" in formatted.columns
    assert "Notes" in formatted.columns
