from __future__ import annotations

from models import MetadataExperimentMethod

METHODS = [
    MetadataExperimentMethod("Query-Aware Top-4", use_metadata_filter=False),
    MetadataExperimentMethod("Query-Aware + Metadata Top-4", use_metadata_filter=True),
]


def should_retrieve(question_type: str) -> bool:
    return question_type.strip().lower() == "book"


def resolve_methods(selected_methods: list[str] | None = None) -> list[MetadataExperimentMethod]:
    if not selected_methods:
        return list(METHODS)
    wanted = {name.strip() for name in selected_methods}
    methods = [method for method in METHODS if method.name in wanted]
    missing = wanted - {method.name for method in methods}
    if missing:
        raise ValueError(f"Unknown method(s): {', '.join(sorted(missing))}")
    return methods
