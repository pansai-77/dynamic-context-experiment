from __future__ import annotations
import random
import time
from dataclasses import asdict
from pathlib import Path
import mlx.core as mx
import numpy as np
import pandas as pd
from .config import Settings
from .index_metadata import verify_index_metadata
from .llm_mlx import QwenMLX
from .models import ExperimentMethod, ExperimentRow
from .prompts import build_prompt
from .results_io import format_detailed_dataframe, filter_questions
from .vector_store import LocalVectorStore

CORE_METHODS = [
    ExperimentMethod("Baseline (Top-8)", 8),
    ExperimentMethod("Standard RAG (Top-4)", 4),
    ExperimentMethod("Minimal RAG (Top-2)", 2),
    ExperimentMethod("No RAG", 0),
    ExperimentMethod("Query-Aware", 4, True),
]

OPTIONAL_METHODS = [
    ExperimentMethod("Query-Aware + Top-2", 2, True),
]

METHODS = CORE_METHODS + OPTIONAL_METHODS

def should_retrieve(question_type: str, method: ExperimentMethod) -> bool:
    if method.name == "No RAG":
        return False
    if method.query_aware:
        return question_type.strip().lower() == "book"
    return method.top_k > 0

def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)

def resolve_methods(
    selected_methods: list[str] | None = None,
    include_optional: bool = False,
) -> list[ExperimentMethod]:
    if selected_methods:
        wanted = {name.strip() for name in selected_methods}
        methods = [method for method in METHODS if method.name in wanted]
        missing = wanted - {method.name for method in methods}
        if missing:
            raise ValueError(f"Unknown method(s): {', '.join(sorted(missing))}")
        return methods
    methods = list(CORE_METHODS)
    if include_optional:
        methods.extend(OPTIONAL_METHODS)
    return methods

def run_experiments(
    settings: Settings,
    limit: int | None = None,
    selected_methods: list[str] | None = None,
    include_optional: bool = False,
    question_ids: list[str] | None = None,
) -> pd.DataFrame:
    set_random_seeds(settings.random_seed)
    questions = filter_questions(pd.read_csv(settings.questions_file), question_ids)
    if limit:
        questions = questions.head(limit)
    methods = resolve_methods(selected_methods, include_optional)

    vector_store = LocalVectorStore(
        settings.qdrant_path, settings.collection_name, settings.embedding_model
    )
    index_metadata = verify_index_metadata(settings)
    print(
        "Index metadata:",
        f"embedding={index_metadata.embedding_model},",
        f"chunk_size={index_metadata.chunk_size},",
        f"chunk_overlap={index_metadata.chunk_overlap},",
        f"sources={index_metadata.source_files}",
    )
    print("Warming up vector store...")
    vector_store.warm_up()
    llm = QwenMLX(settings.llm_model, settings.max_new_tokens, settings.temperature)
    print("Warming up Qwen model...")
    llm.warm_up()

    rows = []
    total_runs = len(questions) * len(methods)
    run_number = 0
    for _, q in questions.iterrows():
        question_id = str(q["Question ID"])
        question_type = str(q["Question Type"])
        question = str(q["Question"])
        for method in methods:
            run_number += 1
            print(f"[{run_number}/{total_runs}] {question_id} - {method.name}")
            total_started = time.perf_counter()
            use_retrieval = should_retrieve(question_type, method)
            effective_top_k = method.top_k if use_retrieval else 0
            retrieved, retrieval_ms = ([], 0.0)
            if use_retrieval:
                retrieved, retrieval_ms = vector_store.search(question, effective_top_k)
            prompt = build_prompt(question, question_type, retrieved)
            generation = llm.answer(prompt)
            total_ms = (time.perf_counter() - total_started) * 1000
            rows.append(ExperimentRow(
                question_id=question_id,
                question_type=question_type,
                question=question,
                method=method.name,
                top_k=effective_top_k,
                used_retrieval=use_retrieval,
                input_tokens=generation.input_tokens,
                output_tokens=generation.output_tokens,
                total_tokens=generation.total_tokens,
                retrieval_time_ms=round(retrieval_ms, 3),
                llm_time_ms=round(generation.llm_time_ms, 3),
                total_time_ms=round(total_ms, 3),
                tokens_per_second=round(generation.tokens_per_second, 3),
                answer=generation.answer,
                retrieved_chunks=len(retrieved),
                retrieved_sources="; ".join(
                    f"p{x.chunk.page_number}:{x.score:.3f}" for x in retrieved
                ),
            ))
    return pd.DataFrame([asdict(row) for row in rows])

def export_detailed_results(dataframe: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    format_detailed_dataframe(dataframe).to_excel(
        output_path, index=False, sheet_name="Detailed Results"
    )
