from __future__ import annotations
from pathlib import Path
import pandas as pd

BASELINE_METHOD = "Baseline (Top-8)"
SCORE_COLUMN = "Score(0-3)"


def normalize_detailed_columns(detailed: pd.DataFrame) -> pd.DataFrame:
    """Accept manual Excel edits such as 'Score (0-3)' with extra spaces."""
    normalized = detailed.copy()
    for name in normalized.columns:
        if name.replace(" ", "") == SCORE_COLUMN:
            if name != SCORE_COLUMN:
                normalized = normalized.rename(columns={name: SCORE_COLUMN})
            return normalized
    raise KeyError(
        f"Missing score column in detailed results. Expected '{SCORE_COLUMN}' "
        f"or a spaced variant such as 'Score (0-3)'."
    )

SUMMARY_COLUMNS_WITH_BASELINE = [
    "Method",
    "Questions",
    "Avg_Input_Tokens",
    "Token Reduction",
    "Avg_Output_Tokens",
    "Avg_Total_Tokens",
    "Avg_Retrieval_Time_ms",
    "Avg_LLM_Time_ms",
    "Avg_Total_Time_ms",
    "Latency Reduction",
    "Avg_Estimated_Cost_USD",
    "Avg_Score",
]

SUMMARY_COLUMNS_WITHOUT_BASELINE = [
    "Method",
    "Questions",
    "Avg_Input_Tokens",
    "Avg_Output_Tokens",
    "Avg_Total_Tokens",
    "Avg_Retrieval_Time_ms",
    "Avg_LLM_Time_ms",
    "Avg_Total_Time_ms",
    "Avg_Estimated_Cost_USD",
    "Avg_Score",
]

def summarise(detailed: pd.DataFrame) -> pd.DataFrame:
    if detailed.empty:
        raise ValueError("Detailed results are empty.")

    summary = detailed.groupby("Method", sort=False).agg(
        Questions=("Question ID", "count"),
        Avg_Input_Tokens=("Input Tokens", "mean"),
        Avg_Output_Tokens=("Output Tokens", "mean"),
        Avg_Total_Tokens=("Total Tokens", "mean"),
        Avg_Retrieval_Time_ms=("Retrieval Time(ms)", "mean"),
        Avg_LLM_Time_ms=("LLM Time(ms)", "mean"),
        Avg_Total_Time_ms=("Total Time(ms)", "mean"),
        Avg_Estimated_Cost_USD=("Estimated Cost(USD)", "mean"),
        Avg_Score=(SCORE_COLUMN, "mean"),
    ).reset_index()

    baseline = detailed[detailed["Method"] == BASELINE_METHOD]
    if baseline.empty:
        return summary[SUMMARY_COLUMNS_WITHOUT_BASELINE]

    baseline_tokens = baseline["Input Tokens"].mean()
    baseline_time = baseline["Total Time(ms)"].mean()
    summary["Token Reduction"] = (baseline_tokens - summary["Avg_Input_Tokens"]) / baseline_tokens
    summary["Latency Reduction"] = (baseline_time - summary["Avg_Total_Time_ms"]) / baseline_time
    return summary[SUMMARY_COLUMNS_WITH_BASELINE]

def create_summary_workbook(detailed_path: Path, summary_path: Path) -> None:
    detailed = normalize_detailed_columns(
        pd.read_excel(detailed_path, sheet_name="Detailed Results")
    )
    with pd.ExcelWriter(summary_path, engine="openpyxl") as writer:
        summarise(detailed).to_excel(writer, index=False, sheet_name="Overall Summary")
        for qtype in ["Book", "General", "Rewrite"]:
            subset = detailed[detailed["Question Type"].str.lower() == qtype.lower()]
            if not subset.empty:
                summarise(subset).to_excel(writer, index=False, sheet_name=f"{qtype} Summary")
