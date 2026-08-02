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
    "Avg QA Retrieval(ms)",
    "Avg LLM Time(ms)",
    "Avg Total Time(ms)",
    "Avg Input Tokens",
    "Avg Output Tokens",
    "Avg Total Tokens",
]

BENCHMARK_TOTAL_COLUMN = "Retrieval Total(ms)"


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


def summarize_benchmark_global(benchmark_runs: pd.DataFrame) -> pd.DataFrame:
    if benchmark_runs.empty:
        return benchmark_runs
    return (
        benchmark_runs.groupby("Method", sort=False)[BENCHMARK_TOTAL_COLUMN]
        .agg(Median="median", P95=lambda values: values.quantile(0.95))
        .reset_index()
    )


def summarize_primary(
    detailed: pd.DataFrame,
    benchmark_runs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    detailed = normalize_detailed_columns(detailed)
    if detailed.empty:
        raise ValueError("Detailed results are empty.")

    benchmark_global = (
        summarize_benchmark_global(benchmark_runs)
        if benchmark_runs is not None and not benchmark_runs.empty
        else None
    )

    rows = []
    for method in detailed["Method"].drop_duplicates():
        method_df = detailed[detailed["Method"] == method]
        book_df = method_df[method_df["Question Type"].str.lower() == "book"]

        median_retrieval = None
        p95_retrieval = None
        if benchmark_global is not None:
            method_benchmark = benchmark_global[benchmark_global["Method"] == method]
            if not method_benchmark.empty:
                median_retrieval = method_benchmark["Median"].iloc[0]
                p95_retrieval = method_benchmark["P95"].iloc[0]

        qa_retrieval_column = "QA Retrieval Time(ms)"
        avg_qa_retrieval = (
            method_df[qa_retrieval_column].mean()
            if qa_retrieval_column in method_df.columns
            else None
        )

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
                "Avg QA Retrieval(ms)": avg_qa_retrieval,
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
    benchmark_path: Path | None = None,
) -> None:
    detailed = normalize_detailed_columns(
        pd.read_excel(detailed_path, sheet_name="Detailed Results")
    )
    benchmark_runs = None
    if benchmark_path is not None and benchmark_path.exists():
        benchmark_runs = pd.read_excel(benchmark_path, sheet_name="Benchmark Runs")
    summary = summarize_primary(detailed, benchmark_runs)
    with pd.ExcelWriter(summary_path, engine="openpyxl") as writer:
        summary.to_excel(writer, index=False, sheet_name="Primary")
