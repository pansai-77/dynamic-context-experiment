from __future__ import annotations

from pathlib import Path

import pandas as pd

SCORE_COLUMN = "Score(0-3)"
BASELINE_METHOD = "Query-Aware Top-4"

TIMING_COLUMNS = [
    "Embed Query Time (ms)",
    "Router Time (ms)",
    "Filter Time (ms)",
    "Vector Search Time (ms)",
    "Search Only Time (ms)",
    "Online Retrieval Time (ms)",
    "Generation Time (ms)",
    "End-to-End Time (ms)",
]

DETAILED_COLUMNS = [
    "Question ID",
    "Question Type",
    "Question",
    "Method",
    "Answer",
    "Retrieved Chunks",
    *TIMING_COLUMNS,
    "Input Tokens",
    "Output Tokens",
    "Total Tokens",
    SCORE_COLUMN,
]

SUMMARY_TIMING_COLUMNS = [
    "Avg Embed Query Time (ms)",
    "Avg Router Time (ms)",
    "Avg Filter Time (ms)",
    "Avg Vector Search Time (ms)",
    "Avg Search Only Time (ms)",
    "Avg Online Retrieval Time (ms)",
    "Avg Generation Time (ms)",
    "Avg End-to-End Time (ms)",
]

SUMMARY_COLUMNS = [
    "Method",
    "Book Score",
    "Overall Score",
    *SUMMARY_TIMING_COLUMNS,
    "Online Retrieval Latency Reduction",
    "End-to-End Latency Reduction",
    "Avg Input Tokens",
    "Avg Output Tokens",
    "Avg Total Tokens",
]

LEGACY_COLUMN_ALIASES = {
    "QA Retrieval Time(ms)": "Search Only Time (ms)",
    "Retrieval Time(ms)": "Search Only Time (ms)",
    "LLM Time(ms)": "Generation Time (ms)",
    "Total Time(ms)": "End-to-End Time (ms)",
}


def _coerce_retrieved_chunks(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric.fillna(0).astype(int)
    return pd.Series([0] * len(series), index=series.index, dtype=int)


def _zero_timing_columns(frame: pd.DataFrame) -> pd.DataFrame:
    for column in TIMING_COLUMNS:
        if column not in frame.columns:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    return frame


def normalize_detailed_columns(detailed: pd.DataFrame) -> pd.DataFrame:
    normalized = detailed.copy()
    normalized = normalized.rename(columns=LEGACY_COLUMN_ALIASES)

    for name in list(normalized.columns):
        if name.replace(" ", "") == SCORE_COLUMN.replace(" ", "") and name != SCORE_COLUMN:
            normalized = normalized.rename(columns={name: SCORE_COLUMN})

    if SCORE_COLUMN not in normalized.columns:
        normalized[SCORE_COLUMN] = None

    normalized = _zero_timing_columns(normalized)

    if "Retrieved Chunks" not in normalized.columns:
        normalized["Retrieved Chunks"] = 0
    else:
        normalized["Retrieved Chunks"] = _coerce_retrieved_chunks(normalized["Retrieved Chunks"])

    non_book = normalized["Question Type"].str.lower().isin(["general", "rewrite"])
    for column in TIMING_COLUMNS:
        if column in {"Generation Time (ms)", "End-to-End Time (ms)"}:
            continue
        normalized.loc[non_book, column] = 0.0
    normalized.loc[non_book, "Retrieved Chunks"] = 0

    return normalized


def migrate_detailed_dataframe(detailed: pd.DataFrame) -> pd.DataFrame:
    normalized = normalize_detailed_columns(detailed)
    for column in DETAILED_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = None
    return normalized[DETAILED_COLUMNS]


def _book_rows(method_df: pd.DataFrame) -> pd.DataFrame:
    return method_df[method_df["Question Type"].str.lower() == "book"]


def _latency_reduction(baseline_time: float, method_time: float) -> float | None:
    if baseline_time == 0:
        return None
    reduction = (baseline_time - method_time) / baseline_time * 100
    if abs(reduction) < 1e-9:
        return 0.0
    return reduction


def summarize_primary(detailed: pd.DataFrame) -> pd.DataFrame:
    detailed = normalize_detailed_columns(detailed)
    if detailed.empty:
        raise ValueError("Detailed results are empty.")

    rows = []
    for method in detailed["Method"].drop_duplicates():
        method_df = detailed[detailed["Method"] == method]
        book_df = _book_rows(method_df)

        book_score = book_df[SCORE_COLUMN].mean() if SCORE_COLUMN in book_df else None
        overall_score = method_df[SCORE_COLUMN].mean() if SCORE_COLUMN in method_df else None
        rows.append(
            {
                "Method": method,
                "Book Score": book_score,
                "Overall Score": overall_score,
                "Avg Embed Query Time (ms)": book_df["Embed Query Time (ms)"].mean(),
                "Avg Router Time (ms)": book_df["Router Time (ms)"].mean(),
                "Avg Filter Time (ms)": book_df["Filter Time (ms)"].mean(),
                "Avg Vector Search Time (ms)": book_df["Vector Search Time (ms)"].mean(),
                "Avg Search Only Time (ms)": book_df["Search Only Time (ms)"].mean(),
                "Avg Online Retrieval Time (ms)": book_df["Online Retrieval Time (ms)"].mean(),
                "Avg Generation Time (ms)": method_df["Generation Time (ms)"].mean(),
                "Avg End-to-End Time (ms)": method_df["End-to-End Time (ms)"].mean(),
                "Online Retrieval Latency Reduction": None,
                "End-to-End Latency Reduction": None,
                "Avg Input Tokens": book_df["Input Tokens"].mean(),
                "Avg Output Tokens": book_df["Output Tokens"].mean(),
                "Avg Total Tokens": book_df["Total Tokens"].mean(),
            }
        )

    summary = pd.DataFrame(rows)
    baseline_rows = summary[summary["Method"] == BASELINE_METHOD]
    if not baseline_rows.empty:
        baseline = baseline_rows.iloc[0]
        baseline_online = float(baseline["Avg Online Retrieval Time (ms)"])
        baseline_end_to_end = float(baseline["Avg End-to-End Time (ms)"])
        summary["Online Retrieval Latency Reduction"] = summary.apply(
            lambda row: _latency_reduction(
                baseline_online,
                float(row["Avg Online Retrieval Time (ms)"]),
            ),
            axis=1,
        )
        summary["End-to-End Latency Reduction"] = summary.apply(
            lambda row: _latency_reduction(
                baseline_end_to_end,
                float(row["Avg End-to-End Time (ms)"]),
            ),
            axis=1,
        )
        summary.loc[summary["Method"] == BASELINE_METHOD, "Online Retrieval Latency Reduction"] = 0.0
        summary.loc[summary["Method"] == BASELINE_METHOD, "End-to-End Latency Reduction"] = 0.0

    return summary[SUMMARY_COLUMNS]


def migrate_detailed_workbook(detailed_path: Path) -> pd.DataFrame:
    detailed = pd.read_excel(detailed_path, sheet_name="Detailed Results")
    migrated = migrate_detailed_dataframe(detailed)
    migrated.to_excel(detailed_path, index=False, sheet_name="Detailed Results")
    return migrated


def create_summary_workbook(detailed_path: Path, summary_path: Path) -> None:
    detailed = migrate_detailed_dataframe(
        pd.read_excel(detailed_path, sheet_name="Detailed Results")
    )
    summary = summarize_primary(detailed)
    with pd.ExcelWriter(summary_path, engine="openpyxl") as writer:
        summary.to_excel(writer, index=False, sheet_name="Primary")
