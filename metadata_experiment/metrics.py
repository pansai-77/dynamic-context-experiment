from __future__ import annotations

METHOD_A = "Query-Aware Top-4"
METHOD_B = "Query-Aware + Metadata Top-4"
METHODS = (METHOD_A, METHOD_B)


def should_retrieve(question_type: str) -> bool:
    return question_type.strip().lower() == "book"
