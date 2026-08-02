from __future__ import annotations

from pathlib import Path

import pandas as pd

SCORE_COLUMN = "Score(0-3)"

DETAILED_COLUMNS = [
    "Question ID",
    "Question Type",
    "Question",
    "Method",
    "Answer",
    "Retrieved Chunks",
    "Retrieval Time(ms)",
    "LLM Time(ms)",
    "Total Time(ms)",
    "Input Tokens",
    "Output Tokens",
    "Total Tokens",
    "Score(0-3)",
]

SUMMARY_COLUMNS = [
    "Method",
    "Book Score",
    "Overall Score",
    "Avg Retrieval Time(ms)",
    "Avg LLM Time(ms)",
    "Avg Total Time(ms)",
    "Avg Input Tokens",
    "Avg Output Tokens",
    "Avg Total Tokens",
]

LEGACY_COLUMN_ALIASES = {
    "QA Retrieval Time(ms)": "Retrieval Time(ms)",
}


def _coerce_retrieved_chunks(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric.fillna(0).astype(int)
    return pd.Series([0] * len(series), index=series.index, dtype=int)


def normalize_detailed_columns(detailed: pd.DataFrame) -> pd.DataFrame:
    normalized = detailed.copy()
    normalized = normalized.rename(columns=LEGACY_COLUMN_ALIASES)

    for name in list(normalized.columns):
        if name.replace(" ", "") == SCORE_COLUMN.replace(" ", "") and name != SCORE_COLUMN:
            normalized = normalized.rename(columns={name: SCORE_COLUMN})

    if SCORE_COLUMN not in normalized.columns:
        normalized[SCORE_COLUMN] = None
    if "Retrieval Time(ms)" not in normalized.columns:
        normalized["Retrieval Time(ms)"] = 0.0
    if "Retrieved Chunks" not in normalized.columns:
        normalized["Retrieved Chunks"] = 0
    else:
        normalized["Retrieved Chunks"] = _coerce_retrieved_chunks(normalized["Retrieved Chunks"])

    non_book = normalized["Question Type"].str.lower().isin(["general", "rewrite"])
    normalized.loc[non_book, "Retrieval Time(ms)"] = 0.0
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
                "Avg Retrieval Time(ms)": book_df["Retrieval Time(ms)"].mean(),
                "Avg LLM Time(ms)": method_df["LLM Time(ms)"].mean(),
                "Avg Total Time(ms)": method_df["Total Time(ms)"].mean(),
                "Avg Input Tokens": book_df["Input Tokens"].mean(),
                "Avg Output Tokens": book_df["Output Tokens"].mean(),
                "Avg Total Tokens": book_df["Total Tokens"].mean(),
            }
        )
    return pd.DataFrame(rows)[SUMMARY_COLUMNS]


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
