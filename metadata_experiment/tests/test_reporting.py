from __future__ import annotations

import pandas as pd
import pytest

from reporting import migrate_detailed_dataframe, normalize_detailed_columns, summarize_primary


def test_migrate_detailed_dataframe_maps_legacy_retrieval_to_search_only():
    detailed = pd.DataFrame(
        {
            "Question ID": ["Q01"],
            "Question Type": ["book"],
            "Question": ["test"],
            "Method": ["Query-Aware Top-4"],
            "QA Retrieval Time(ms)": [12.0],
            "LLM Time(ms)": [100.0],
            "Total Time(ms)": [112.0],
            "Input Tokens": [1000],
            "Output Tokens": [20],
            "Total Tokens": [1020],
            "Answer": ["answer"],
            "Score(0-3)": [2],
        }
    )
    migrated = migrate_detailed_dataframe(detailed)
    assert migrated.loc[0, "Search Only Time (ms)"] == 12.0
    assert migrated.loc[0, "Generation Time (ms)"] == 100.0
    assert migrated.loc[0, "End-to-End Time (ms)"] == 112.0
    assert migrated.loc[0, "Retrieved Chunks"] == 0


def test_normalize_detailed_columns_zeros_non_book_retrieval_fields():
    detailed = pd.DataFrame(
        {
            "Question ID": ["Q11"],
            "Question Type": ["general"],
            "Question": ["test"],
            "Method": ["Query-Aware Top-4"],
            "Answer": ["answer"],
            "Embed Query Time (ms)": [69.0],
            "Search Only Time (ms)": [69.0],
            "Retrieved Chunks": [69],
            "Generation Time (ms)": [100.0],
            "End-to-End Time (ms)": [100.0],
            "Input Tokens": [100],
            "Output Tokens": [10],
            "Total Tokens": [110],
        }
    )
    normalized = normalize_detailed_columns(detailed)
    assert normalized.loc[0, "Embed Query Time (ms)"] == 0.0
    assert normalized.loc[0, "Search Only Time (ms)"] == 0.0
    assert normalized.loc[0, "Retrieved Chunks"] == 0
    assert normalized.loc[0, "Generation Time (ms)"] == 100.0


def test_summarize_primary_accepts_legacy_retrieval_column_name():
    detailed = pd.DataFrame(
        {
            "Method": ["Query-Aware Top-4"],
            "Question Type": ["book"],
            "QA Retrieval Time(ms)": [12.0],
            "Embed Query Time (ms)": [50.0],
            "Online Retrieval Time (ms)": [62.0],
            "Generation Time (ms)": [100.0],
            "End-to-End Time (ms)": [112.0],
            "Input Tokens": [1000],
            "Output Tokens": [20],
            "Total Tokens": [1020],
        }
    )
    summary = summarize_primary(detailed)
    assert summary.loc[0, "Avg Search Only Time (ms)"] == 12.0
    assert summary.loc[0, "Avg Online Retrieval Time (ms)"] == 62.0


def test_summarize_primary_computes_latency_reduction():
    detailed = pd.DataFrame(
        {
            "Method": ["Query-Aware Top-4", "Query-Aware + Metadata Top-4"],
            "Question Type": ["book", "book"],
            "Embed Query Time (ms)": [100.0, 100.0],
            "Router Time (ms)": [0.0, 0.01],
            "Filter Time (ms)": [0.0, 0.02],
            "Vector Search Time (ms)": [10.0, 8.0],
            "Search Only Time (ms)": [10.0, 8.03],
            "Online Retrieval Time (ms)": [110.0, 108.03],
            "Generation Time (ms)": [100.0, 95.0],
            "End-to-End Time (ms)": [110.0, 103.03],
            "Input Tokens": [1000, 900],
            "Output Tokens": [20, 20],
            "Total Tokens": [1020, 920],
        }
    )
    summary = summarize_primary(detailed)
    baseline = summary[summary["Method"] == "Query-Aware Top-4"].iloc[0]
    metadata = summary[summary["Method"] == "Query-Aware + Metadata Top-4"].iloc[0]
    assert baseline["Online Retrieval Latency Reduction"] == 0.0
    assert metadata["Online Retrieval Latency Reduction"] == pytest.approx(1.7909, rel=1e-3)
