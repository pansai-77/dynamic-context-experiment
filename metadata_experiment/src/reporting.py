from __future__ import annotations

from pathlib import Path

import pandas as pd

SCORE_COLUMN = "Score(0-3)"
METHOD_A = "Query-Aware Top-4"

PRIMARY_COLUMNS = [
    "Method",
    "Questions",
    "Book Score",
    "Overall Score",
    "Median Retrieval(ms)",
    "P95 Retrieval(ms)",
    "Avg LLM Time(ms)",
    "Avg Total Time(ms)",
    "Avg Input Tokens",
    "Avg Output Tokens",
    "Avg Total Tokens",
]


def normalize_detailed_columns(detailed: pd.DataFrame) -> pd.DataFrame:
    normalized = detailed.copy()
    for name in normalized.columns:
        if name.replace(" ", "") == SCORE_COLUMN:
            if name != SCORE_COLUMN:
                normalized = normalized.rename(columns={name: SCORE_COLUMN})
            return normalized
    if SCORE_COLUMN not in normalized.columns:
        normalized[SCORE_COLUMN] = None
    return normalized


def summarize_primary(
    detailed: pd.DataFrame,
    benchmark_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    detailed = normalize_detailed_columns(detailed)
    if detailed.empty:
        raise ValueError("Detailed results are empty.")

    rows = []
    for method in detailed["Method"].drop_duplicates():
        method_df = detailed[detailed["Method"] == method]
        book_df = method_df[method_df["Question Type"].str.lower() == "book"]
        median_retrieval = method_df["Retrieval Time(ms)"].median()
        p95_retrieval = method_df["Retrieval Time(ms)"].quantile(0.95)
        if benchmark_summary is not None and not benchmark_summary.empty:
            method_benchmark = benchmark_summary[benchmark_summary["Method"] == method]
            if not method_benchmark.empty:
                median_retrieval = method_benchmark["Median"].median()
                p95_retrieval = method_benchmark["P95"].median()

        book_score = book_df[SCORE_COLUMN].mean() if SCORE_COLUMN in book_df else None
        overall_score = method_df[SCORE_COLUMN].mean() if SCORE_COLUMN in method_df else None
        rows.append(
            {
                "Method": method,
                "Questions": len(method_df),
                "Book Score": book_score,
                "Overall Score": overall_score,
                "Median Retrieval(ms)": median_retrieval,
                "P95 Retrieval(ms)": p95_retrieval,
                "Avg LLM Time(ms)": method_df["LLM Time(ms)"].mean(),
                "Avg Total Time(ms)": method_df["Total Time(ms)"].mean(),
                "Avg Input Tokens": method_df["Input Tokens"].mean(),
                "Avg Output Tokens": method_df["Output Tokens"].mean(),
                "Avg Total Tokens": method_df["Total Tokens"].mean(),
            }
        )
    return pd.DataFrame(rows)[PRIMARY_COLUMNS]


def create_summary_workbook(
    detailed_path: Path,
    summary_path: Path,
    benchmark_summary_path: Path | None = None,
) -> None:
    detailed = normalize_detailed_columns(
        pd.read_excel(detailed_path, sheet_name="Detailed Results")
    )
    benchmark_summary = None
    if benchmark_summary_path is not None and benchmark_summary_path.exists():
        benchmark_summary = pd.read_excel(benchmark_summary_path, sheet_name="Benchmark Summary")
    summary = summarize_primary(detailed, benchmark_summary)
    with pd.ExcelWriter(summary_path, engine="openpyxl") as writer:
        summary.to_excel(writer, index=False, sheet_name="Primary")
