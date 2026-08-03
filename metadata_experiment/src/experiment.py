from __future__ import annotations

import json
import random
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config import MetadataSettings, settings
from index_metadata import verify_index_metadata
from logic import resolve_methods, should_retrieve
from metadata_retriever import MetadataVectorStore, build_router_from_settings
from models import MetadataExperimentMethod, MetadataExperimentRow, RetrievalTiming
from src.llm_mlx import QwenMLX
from src.prompts import build_prompt
from src.results_io import filter_questions
from reporting import migrate_detailed_dataframe


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import mlx.core as mx

    mx.random.seed(seed)


def run_qa_experiments(
    cfg: MetadataSettings,
    vector_store: MetadataVectorStore,
    router,
    llm: QwenMLX,
    questions: pd.DataFrame,
    methods: list[MetadataExperimentMethod],
    allow_topic_expansion: bool = False,
) -> pd.DataFrame:
    rows: list[MetadataExperimentRow] = []
    total_runs = len(questions) * len(methods)
    run_number = 0
    for _, question_row in questions.iterrows():
        question_id = str(question_row["Question ID"])
        question_type = str(question_row["Question Type"])
        question = str(question_row["Question"])
        use_retrieval = should_retrieve(question_type)
        embed_query_ms = 0.0
        query_vector = None
        if use_retrieval:
            embed_started = time.perf_counter()
            query_vector = vector_store.embed_query(question)
            embed_query_ms = (time.perf_counter() - embed_started) * 1000

        for method in methods:
            run_number += 1
            print(f"[{run_number}/{total_runs}] {question_id} - {method.name}")
            retrieved = []
            timing = RetrievalTiming()
            topic_ids: list[str] = []
            if use_retrieval and query_vector is not None:
                if method.use_metadata_filter:
                    retrieved, timing, topic_ids = vector_store.search_with_metadata(
                        query_vector,
                        router,
                        top_k=cfg.top_k,
                        allow_topic_expansion=allow_topic_expansion,
                    )
                else:
                    retrieved, timing = vector_store.search_full(query_vector, top_k=cfg.top_k)

                if method.use_metadata_filter and not retrieved:
                    print(
                        f"  Warning: {question_id} metadata filter returned 0 chunks "
                        f"(topics={topic_ids})"
                    )

            router_ms = timing.router_time_ms
            filter_ms = timing.filter_build_time_ms
            vector_search_ms = timing.vector_search_time_ms
            search_only_ms = timing.search_only_ms
            online_retrieval_ms = embed_query_ms + search_only_ms

            prompt = build_prompt(question, question_type, retrieved)
            generation = llm.answer(prompt)
            generation_ms = generation.llm_time_ms
            end_to_end_ms = online_retrieval_ms + generation_ms
            rows.append(
                MetadataExperimentRow(
                    question_id=question_id,
                    question_type=question_type,
                    question=question,
                    method=method.name,
                    retrieved_chunks=len(retrieved),
                    input_tokens=generation.input_tokens,
                    output_tokens=generation.output_tokens,
                    total_tokens=generation.total_tokens,
                    embed_query_time_ms=round(embed_query_ms, 3),
                    router_time_ms=round(router_ms, 3),
                    filter_time_ms=round(filter_ms, 3),
                    vector_search_time_ms=round(vector_search_ms, 3),
                    search_only_time_ms=round(search_only_ms, 3),
                    online_retrieval_time_ms=round(online_retrieval_ms, 3),
                    generation_time_ms=round(generation_ms, 3),
                    end_to_end_time_ms=round(end_to_end_ms, 3),
                    answer=generation.answer,
                )
            )
    return pd.DataFrame([asdict(row) for row in rows])


def format_detailed_dataframe(detailed: pd.DataFrame) -> pd.DataFrame:
    formatted = detailed.copy()
    rename_map = {
        "question_id": "Question ID",
        "question_type": "Question Type",
        "question": "Question",
        "method": "Method",
        "retrieved_chunks": "Retrieved Chunks",
        "input_tokens": "Input Tokens",
        "output_tokens": "Output Tokens",
        "total_tokens": "Total Tokens",
        "embed_query_time_ms": "Embed Query Time (ms)",
        "router_time_ms": "Router Time (ms)",
        "filter_time_ms": "Filter Time (ms)",
        "vector_search_time_ms": "Vector Search Time (ms)",
        "search_only_time_ms": "Search Only Time (ms)",
        "online_retrieval_time_ms": "Online Retrieval Time (ms)",
        "generation_time_ms": "Generation Time (ms)",
        "end_to_end_time_ms": "End-to-End Time (ms)",
        "answer": "Answer",
        "score_0_3": "Score(0-3)",
    }
    formatted = formatted.rename(columns=rename_map)
    if "Score(0-3)" not in formatted.columns:
        formatted["Score(0-3)"] = None
    return migrate_detailed_dataframe(formatted)


def create_run_directory(results_dir: Path, run_at: datetime | None = None) -> Path:
    timestamp = (run_at or datetime.now()).strftime("%Y%m%d_%H%M%S")
    run_dir = results_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def export_run_config(
    cfg: MetadataSettings,
    methods: list[str],
    output_path: Path,
    run_directory: Path,
    allow_topic_expansion: bool = False,
) -> None:
    payload = {
        "run_timestamp_utc": datetime.now(UTC).isoformat(),
        "run_directory": str(run_directory),
        "experiment": "metadata_rag_phase_1",
        "methods": methods,
        "settings": {
            "collection_name": cfg.collection_name,
            "embedding_model": cfg.embedding_model,
            "llm_model": cfg.llm_model,
            "chunk_strategy": cfg.chunk_strategy,
            "target_size": cfg.chunk_target_size,
            "max_size": cfg.chunk_max_size,
            "min_size": cfg.chunk_min_size,
            "overlap": cfg.chunk_overlap,
            "top_k": cfg.top_k,
            "topic_routing_top_n": cfg.topic_routing_top_n,
            "allow_topic_expansion": allow_topic_expansion,
            "random_seed": cfg.random_seed,
        },
        "timing_notes": (
            "Timing definitions per method row: Search Only Time (ms) = Router + Filter "
            "Build + Vector Search; Online Retrieval Time (ms) = Embed Query + Search Only; "
            "End-to-End Time (ms) = Online Retrieval + Generation. For each Book question, "
            "query embedding is computed once and shared by both methods to ensure identical "
            "vector input; the same Embed Query Time is assigned to both methods as their "
            "independent deployment cost. Filter Time (ms) records Qdrant Filter object "
            "construction only; filtering executes inside Vector Search Time (ms)."
        ),
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_metadata_experiment(
    cfg: MetadataSettings = settings,
    question_ids: list[str] | None = None,
    selected_methods: list[str] | None = None,
    allow_topic_expansion: bool = False,
) -> pd.DataFrame:
    set_random_seeds(cfg.random_seed)
    questions = filter_questions(pd.read_csv(cfg.questions_file), question_ids)
    methods = resolve_methods(selected_methods)

    verify_index_metadata(cfg)
    vector_store = MetadataVectorStore(cfg.qdrant_path, cfg.collection_name, cfg.embedding_model)
    router = build_router_from_settings(
        vector_store,
        cfg.allowed_topics_file,
        cfg.topic_embeddings_file,
        cfg.topic_routing_top_n,
        cfg.embedding_model,
    )

    print("Warming up vector store...")
    vector_store.warm_up()
    llm = QwenMLX(cfg.llm_model, cfg.max_new_tokens, cfg.temperature)
    print("Warming up LLM...")
    llm.warm_up()

    print("Running QA experiments...")
    return run_qa_experiments(
        cfg,
        vector_store,
        router,
        llm,
        questions,
        methods,
        allow_topic_expansion=allow_topic_expansion,
    )
