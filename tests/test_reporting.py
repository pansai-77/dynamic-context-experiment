import pandas as pd
from src.reporting import BASELINE_METHOD, normalize_detailed_columns, summarise

def _sample_row(method: str, question_id: str = "Q01", question_type: str = "Book") -> dict:
    return {
        "Question ID": question_id,
        "Question Type": question_type,
        "Method": method,
        "Input Tokens": 1000,
        "Output Tokens": 50,
        "Total Tokens": 1050,
        "Retrieval Time(ms)": 100.0,
        "LLM Time(ms)": 2000.0,
        "Total Time(ms)": 2100.0,
        "Score(0-3)": None,
    }

def test_summarise_includes_reductions_when_baseline_present():
    detailed = pd.DataFrame([
        _sample_row(BASELINE_METHOD),
        _sample_row("Query-Aware + Top-2"),
    ])
    summary = summarise(detailed)
    assert "Token Reduction" in summary.columns
    assert "Latency Reduction" in summary.columns
    assert "Avg_Estimated_Cost_USD" not in summary.columns

def test_normalize_detailed_columns_accepts_spaced_score_header():
    detailed = pd.DataFrame([
        {
            "Question ID": "Q01",
            "Question Type": "Book",
            "Method": BASELINE_METHOD,
            "Input Tokens": 1000,
            "Output Tokens": 50,
            "Total Tokens": 1050,
            "Retrieval Time(ms)": 100.0,
            "LLM Time(ms)": 2000.0,
            "Total Time(ms)": 2100.0,
            "Score (0-3)": 3,
        }
    ])
    normalized = normalize_detailed_columns(detailed)
    assert "Score(0-3)" in normalized.columns
    assert "Score (0-3)" not in normalized.columns


def test_summarise_works_without_baseline_for_single_method_run():
    detailed = pd.DataFrame([_sample_row("Query-Aware + Top-2")])
    summary = summarise(detailed)
    assert len(summary) == 1
    assert summary.iloc[0]["Method"] == "Query-Aware + Top-2"
    assert "Token Reduction" not in summary.columns
    assert "Latency Reduction" not in summary.columns
    assert summary.iloc[0]["Avg_Input_Tokens"] == 1000
