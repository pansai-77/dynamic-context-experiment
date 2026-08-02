from __future__ import annotations

import pandas as pd

from reporting import summarize_benchmark_global, summarize_primary


def test_summarize_benchmark_global_uses_all_runs():
    benchmark_runs = pd.DataFrame(
        {
            "Question ID": ["Q01", "Q01", "Q02", "Q02"],
            "Method": ["A", "A", "A", "A"],
            "Retrieval Total(ms)": [10.0, 30.0, 20.0, 40.0],
        }
    )
    summary = summarize_benchmark_global(benchmark_runs)
    assert summary.loc[0, "Median"] == 25.0
    assert summary.loc[0, "P95"] == 38.5


def test_summarize_primary_uses_global_benchmark_stats():
    detailed = pd.DataFrame(
        {
            "Method": ["A", "A"],
            "Question Type": ["book", "general"],
            "QA Retrieval Time(ms)": [12.0, 0.0],
            "LLM Time(ms)": [100.0, 90.0],
            "Total Time(ms)": [112.0, 90.0],
            "Input Tokens": [100, 50],
            "Output Tokens": [20, 10],
            "Total Tokens": [120, 60],
        }
    )
    benchmark_runs = pd.DataFrame(
        {
            "Question ID": ["Q01", "Q01"],
            "Method": ["A", "A"],
            "Retrieval Total(ms)": [10.0, 30.0],
        }
    )
    summary = summarize_primary(detailed, benchmark_runs)
    assert summary.loc[0, "Median Retrieval(ms)"] == 20.0
    assert summary.loc[0, "P95 Retrieval(ms)"] == 29.0
    assert summary.loc[0, "Avg QA Retrieval(ms)"] == 6.0
