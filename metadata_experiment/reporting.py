from __future__ import annotations

from pathlib import Path

import pandas as pd


COLUMN_NAMES = {
    "question_id": "Question ID", "question_type": "Question Type", "question": "Question",
    "method": "Method", "top_k": "Top-k", "used_retrieval": "Used Retrieval",
    "routed_topics": "Routed Topics", "candidates_before_filter": "Candidates Before Filter",
    "candidates_after_filter": "Candidates After Filter", "router_time_ms": "Router Time(ms)",
    "vector_time_ms": "Vector Time(ms)", "retrieval_time_ms": "Retrieval Time(ms)",
    "input_tokens": "Input Tokens", "output_tokens": "Output Tokens", "total_tokens": "Total Tokens",
    "llm_time_ms": "LLM Time(ms)", "total_time_ms": "Total Time(ms)", "answer": "Answer",
    "retrieved_chunks": "Retrieved Chunks", "retrieved_sources": "Retrieved Sources",
    "hit_at_4": "Hit@4", "mrr_at_4": "MRR@4", "filter_accuracy": "Filter Accuracy",
    "score_0_3": "Score(0-3)", "notes": "Notes",
}


def export_detailed(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.rename(columns=COLUMN_NAMES).to_excel(path, index=False, sheet_name="Detailed Results")


def summarise(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby("Method", sort=False).agg(
        Questions=("Question ID", "count"),
        Avg_Input_Tokens=("Input Tokens", "mean"),
        Avg_Retrieval_Time_ms=("Retrieval Time(ms)", "mean"),
        Avg_LLM_Time_ms=("LLM Time(ms)", "mean"),
        Avg_Total_Time_ms=("Total Time(ms)", "mean"),
        Avg_Score=("Score(0-3)", "mean"),
    ).reset_index()


def create_summary(detailed_path: Path, summary_path: Path) -> None:
    detailed = pd.read_excel(detailed_path, sheet_name="Detailed Results")
    if "Score(0-3)" not in detailed:
        detailed["Score(0-3)"] = None
    with pd.ExcelWriter(summary_path, engine="openpyxl") as writer:
        summarise(detailed).to_excel(writer, index=False, sheet_name="Overall Summary")
        for qtype in ("Book", "General", "Rewrite"):
            subset = detailed[detailed["Question Type"].str.lower() == qtype.lower()]
            if not subset.empty:
                summarise(subset).to_excel(writer, index=False, sheet_name=f"{qtype} Summary")
