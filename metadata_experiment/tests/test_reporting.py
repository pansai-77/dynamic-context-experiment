from __future__ import annotations

import pandas as pd

from reporting import migrate_detailed_dataframe, normalize_detailed_columns, summarize_primary


def test_migrate_detailed_dataframe_drops_legacy_columns():
    detailed = pd.DataFrame(
        {
            "Question ID": ["Q01"],
            "Question Type": ["book"],
            "Question": ["test"],
            "Method": ["A"],
            "Top-k": [4],
            "Used Retrieval": [True],
            "QA Retrieval Time(ms)": [12.0],
            "Benchmark Median Retrieval(ms)": [10.0],
            "LLM Time(ms)": [100.0],
            "Total Time(ms)": [112.0],
            "Input Tokens": [1000],
            "Output Tokens": [20],
            "Total Tokens": [1020],
            "Output Tokens/sec": [5.0],
            "Answer": ["answer"],
            "Retrieved Chunk IDs": ["p001-c001"],
            "Retrieved Sources": ["p1:0.9"],
            "Score(0-3)": [2],
        }
    )
    migrated = migrate_detailed_dataframe(detailed)
    assert "Retrieved Chunk IDs" not in migrated.columns
    assert migrated.loc[0, "Retrieval Time(ms)"] == 12.0
    assert migrated.loc[0, "Retrieved Chunks"] == 0


def test_normalize_detailed_columns_zeros_non_book_retrieval_fields():
    detailed = pd.DataFrame(
        {
            "Question ID": ["Q11"],
            "Question Type": ["general"],
            "Question": ["test"],
            "Method": ["A"],
            "Answer": ["answer"],
            "Retrieval Time(ms)": [69.0],
            "Retrieved Chunks": [69],
            "LLM Time(ms)": [100.0],
            "Total Time(ms)": [100.0],
            "Input Tokens": [100],
            "Output Tokens": [10],
            "Total Tokens": [110],
        }
    )
    normalized = normalize_detailed_columns(detailed)
    assert normalized.loc[0, "Retrieval Time(ms)"] == 0.0
    assert normalized.loc[0, "Retrieved Chunks"] == 0


def test_summarize_primary_accepts_legacy_retrieval_column_name():
    detailed = pd.DataFrame(
        {
            "Method": ["A"],
            "Question Type": ["book"],
            "QA Retrieval Time(ms)": [12.0],
            "LLM Time(ms)": [100.0],
            "Total Time(ms)": [112.0],
            "Input Tokens": [1000],
            "Output Tokens": [20],
            "Total Tokens": [1020],
        }
    )
    summary = summarize_primary(detailed)
    assert summary.loc[0, "Avg Retrieval Time(ms)"] == 12.0


def test_summarize_primary_matches_guide_metrics():
    detailed = pd.DataFrame(
        {
            "Method": ["A", "A"],
            "Question Type": ["book", "general"],
            "Retrieval Time(ms)": [12.0, 0.0],
            "LLM Time(ms)": [100.0, 90.0],
            "Total Time(ms)": [112.0, 90.0],
            "Input Tokens": [1000, 100],
            "Output Tokens": [20, 10],
            "Total Tokens": [1020, 110],
        }
    )
    summary = summarize_primary(detailed)
    assert summary.loc[0, "Avg Retrieval Time(ms)"] == 12.0
    assert summary.loc[0, "Avg Input Tokens"] == 1000.0
    assert summary.loc[0, "Avg LLM Time(ms)"] == 95.0
