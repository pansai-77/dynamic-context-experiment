from __future__ import annotations

from pathlib import Path

import pandas as pd

from .reporting import normalize_detailed_columns

DETAILED_COLUMN_RENAMES = {
    "question_id": "Question ID",
    "question_type": "Question Type",
    "question": "Question",
    "method": "Method",
    "top_k": "Top-k",
    "used_retrieval": "Used Retrieval",
    "input_tokens": "Input Tokens",
    "output_tokens": "Output Tokens",
    "total_tokens": "Total Tokens",
    "retrieval_time_ms": "Retrieval Time(ms)",
    "llm_time_ms": "LLM Time(ms)",
    "total_time_ms": "Total Time(ms)",
    "tokens_per_second": "Output Tokens/sec",
    "answer": "Answer",
    "retrieved_chunks": "Retrieved Chunks",
    "retrieved_sources": "Retrieved Sources",
    "score_0_3": "Score(0-3)",
    "notes": "Notes",
}

def parse_question_ids(question_ids: str | None) -> list[str] | None:
    if not question_ids:
        return None
    parsed = [part.strip() for part in question_ids.split(",") if part.strip()]
    return parsed or None

def filter_questions(questions: pd.DataFrame, question_ids: list[str] | None) -> pd.DataFrame:
    if not question_ids:
        return questions
    filtered = questions[questions["Question ID"].isin(question_ids)].copy()
    missing = sorted(set(question_ids) - set(filtered["Question ID"]))
    if missing:
        raise ValueError(f"Unknown question ID(s): {', '.join(missing)}")
    order = {question_id: index for index, question_id in enumerate(question_ids)}
    filtered["_question_order"] = filtered["Question ID"].map(order)
    return filtered.sort_values("_question_order").drop(columns="_question_order")

def format_detailed_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    exported = dataframe.rename(columns=DETAILED_COLUMN_RENAMES)
    if "Score(0-3)" not in exported.columns:
        exported["Score(0-3)"] = None
    if "Notes" not in exported.columns:
        exported["Notes"] = ""
    return exported

def patch_detailed_results(existing_path: Path, new_dataframe: pd.DataFrame) -> pd.DataFrame:
    existing = normalize_detailed_columns(
        pd.read_excel(existing_path, sheet_name="Detailed Results")
    )
    incoming = format_detailed_dataframe(new_dataframe)
    incoming["Score(0-3)"] = None
    incoming["Notes"] = ""

    incoming_lookup = {
        (str(row["Question ID"]), str(row["Method"])): row.to_dict()
        for _, row in incoming.iterrows()
    }
    expected_keys = set(incoming_lookup)
    rows: list[dict] = []
    applied_keys: set[tuple[str, str]] = set()

    for _, row in existing.iterrows():
        key = (str(row["Question ID"]), str(row["Method"]))
        if key in incoming_lookup:
            rows.append(incoming_lookup[key])
            applied_keys.add(key)
        else:
            rows.append(row.to_dict())

    missing_keys = expected_keys - applied_keys
    if missing_keys:
        formatted = ", ".join(
            f"{question_id}/{method}" for question_id, method in sorted(missing_keys)
        )
        raise ValueError(
            f"Patch targets not found in existing detailed results: {formatted}"
        )
    return pd.DataFrame(rows)
