from __future__ import annotations

import json
import random
import time
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config import MetadataSettings, settings
from index_metadata import verify_index_metadata
from logic import METHODS, resolve_methods, should_retrieve
from metadata_retriever import MetadataVectorStore, build_router_from_settings
from models import MetadataExperimentMethod, MetadataExperimentRow
from src.llm_mlx import QwenMLX
from src.prompts import build_prompt
from src.results_io import filter_questions

METHOD_A = METHODS[0].name
METHOD_B = METHODS[1].name


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import mlx.core as mx

    mx.random.seed(seed)


def _format_chunk_ids(retrieved) -> str:
    return "|".join(item.chunk.chunk_id for item in retrieved)


def _format_sources(retrieved) -> str:
    return "; ".join(f"p{item.chunk.page_number}:{item.score:.3f}" for item in retrieved)


def run_retrieval_benchmark(
    cfg: MetadataSettings,
    vector_store: MetadataVectorStore,
    router,
    questions: pd.DataFrame,
    methods: list[MetadataExperimentMethod],
) -> pd.DataFrame:
    book_questions = questions[questions["Question Type"].str.lower() == "book"]
    rows = []
    for _, question_row in book_questions.iterrows():
        question_id = str(question_row["Question ID"])
        question = str(question_row["Question"])
        query_vector = vector_store.embed_query(question)
        for repeat_index in range(1, cfg.benchmark_repeats + 1):
            order = METHODS if repeat_index % 2 == 1 else list(reversed(METHODS))
            for method in order:
                if method.name not in {item.name for item in methods}:
                    continue
                if method.use_metadata_filter:
                    _, timing, _ = vector_store.search_with_metadata(
                        query_vector,
                        router,
                        top_k=cfg.top_k,
                    )
                else:
                    _, timing = vector_store.search_full(query_vector, top_k=cfg.top_k)
                rows.append(
                    {
                        "Question ID": question_id,
                        "Method": method.name,
                        "Repeat Index": repeat_index,
                        "Router Time(ms)": round(timing.router_time_ms, 3),
                        "Filter Build Time(ms)": round(timing.filter_build_time_ms, 3),
                        "Vector Search Time(ms)": round(timing.vector_search_time_ms, 3),
                        "Retrieval Total(ms)": round(timing.retrieval_total_ms, 3),
                    }
                )
    return pd.DataFrame(rows)


def summarize_benchmark(benchmark_df: pd.DataFrame) -> pd.DataFrame:
    if benchmark_df.empty:
        return benchmark_df
    summary = (
        benchmark_df.groupby(["Question ID", "Method"], sort=False)["Retrieval Total(ms)"]
        .agg(Median="median", Mean="mean", P95=lambda values: values.quantile(0.95))
        .reset_index()
    )
    return summary


def run_qa_experiments(
    cfg: MetadataSettings,
    vector_store: MetadataVectorStore,
    router,
    llm: QwenMLX,
    questions: pd.DataFrame,
    methods: list[MetadataExperimentMethod],
    benchmark_medians: pd.DataFrame | None = None,
) -> pd.DataFrame:
    median_lookup: dict[tuple[str, str], float] = {}
    if benchmark_medians is not None and not benchmark_medians.empty:
        for _, row in benchmark_medians.iterrows():
            median_lookup[(str(row["Question ID"]), str(row["Method"]))] = float(row["Median"])

    rows: list[MetadataExperimentRow] = []
    total_runs = len(questions) * len(methods)
    run_number = 0
    for _, question_row in questions.iterrows():
        question_id = str(question_row["Question ID"])
        question_type = str(question_row["Question Type"])
        question = str(question_row["Question"])
        use_retrieval = should_retrieve(question_type)
        query_vector = vector_store.embed_query(question) if use_retrieval else None

        for method in methods:
            run_number += 1
            print(f"[{run_number}/{total_runs}] {question_id} - {method.name}")
            total_started = time.perf_counter()
            effective_top_k = cfg.top_k if use_retrieval else 0
            retrieved = []
            qa_retrieval_ms = 0.0
            benchmark_median_ms: float | None = None
            if use_retrieval and query_vector is not None:
                if method.use_metadata_filter:
                    retrieved, timing, _ = vector_store.search_with_metadata(
                        query_vector,
                        router,
                        top_k=cfg.top_k,
                    )
                    qa_retrieval_ms = timing.retrieval_total_ms
                else:
                    retrieved, timing = vector_store.search_full(query_vector, top_k=cfg.top_k)
                    qa_retrieval_ms = timing.retrieval_total_ms

                benchmark_key = (question_id, method.name)
                if benchmark_key in median_lookup:
                    benchmark_median_ms = median_lookup[benchmark_key]

            prompt = build_prompt(question, question_type, retrieved)
            generation = llm.answer(prompt)
            total_ms = (time.perf_counter() - total_started) * 1000
            rows.append(
                MetadataExperimentRow(
                    question_id=question_id,
                    question_type=question_type,
                    question=question,
                    method=method.name,
                    top_k=effective_top_k,
                    used_retrieval=use_retrieval,
                    input_tokens=generation.input_tokens,
                    output_tokens=generation.output_tokens,
                    total_tokens=generation.total_tokens,
                    qa_retrieval_time_ms=round(qa_retrieval_ms, 3),
                    benchmark_median_retrieval_ms=(
                        round(benchmark_median_ms, 3) if benchmark_median_ms is not None else None
                    ),
                    llm_time_ms=round(generation.llm_time_ms, 3),
                    total_time_ms=round(total_ms, 3),
                    tokens_per_second=round(generation.tokens_per_second, 3),
                    answer=generation.answer,
                    retrieved_chunk_ids=_format_chunk_ids(retrieved),
                    retrieved_sources=_format_sources(retrieved),
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
        "top_k": "Top-k",
        "used_retrieval": "Used Retrieval",
        "input_tokens": "Input Tokens",
        "output_tokens": "Output Tokens",
        "total_tokens": "Total Tokens",
        "qa_retrieval_time_ms": "QA Retrieval Time(ms)",
        "benchmark_median_retrieval_ms": "Benchmark Median Retrieval(ms)",
        "llm_time_ms": "LLM Time(ms)",
        "total_time_ms": "Total Time(ms)",
        "tokens_per_second": "Output Tokens/sec",
        "answer": "Answer",
        "retrieved_chunk_ids": "Retrieved Chunk IDs",
        "retrieved_sources": "Retrieved Sources",
        "score_0_3": "Score(0-3)",
    }
    formatted = formatted.rename(columns=rename_map)
    if "Score(0-3)" not in formatted.columns:
        formatted["Score(0-3)"] = None
    return formatted


def create_scoring_sheet(detailed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    formatted = format_detailed_dataframe(detailed)
    mapping_rows = []
    scoring_rows = []
    for _, row in formatted.iterrows():
        random_row_id = f"S{uuid.uuid4().hex[:8].upper()}"
        mapping_rows.append(
            {
                "Random Row ID": random_row_id,
                "Question ID": row["Question ID"],
                "Method": row["Method"],
            }
        )
        scoring_rows.append(
            {
                "Random Row ID": random_row_id,
                "Question": row["Question"],
                "Answer": row["Answer"],
                "Score(0-3)": row.get("Score(0-3)"),
            }
        )
    scoring_df = pd.DataFrame(scoring_rows).sample(frac=1.0, random_state=settings.random_seed).reset_index(drop=True)
    mapping_df = pd.DataFrame(mapping_rows)
    return scoring_df, mapping_df


def merge_scoring_sheet(detailed_path: Path, scoring_path: Path, mapping_path: Path) -> pd.DataFrame:
    detailed = format_detailed_dataframe(pd.read_excel(detailed_path, sheet_name="Detailed Results"))
    scoring = pd.read_excel(scoring_path)
    mapping = pd.read_csv(mapping_path)
    merged = mapping.merge(scoring, on="Random Row ID", how="left")
    score_map = {
        (row["Question ID"], row["Method"]): row["Score(0-3)"]
        for _, row in merged.iterrows()
    }
    detailed["Score(0-3)"] = detailed.apply(
        lambda row: score_map.get((row["Question ID"], row["Method"])),
        axis=1,
    )
    return detailed


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
) -> None:
    payload = {
        "run_timestamp_utc": datetime.now(UTC).isoformat(),
        "run_directory": str(run_directory),
        "experiment": "metadata_rag_phase_1",
        "methods": methods,
        "timing_notes": {
            "total_time_scope": "Per QA run from retrieval through generation; excludes the shared query-embedding step computed once per question before method loops.",
            "qa_retrieval_time_scope": "Book questions only in summary averages; General/Rewrite rows remain 0 because retrieval is skipped by design.",
            "benchmark_retrieval_scope": "Independent 25-repeat benchmark on Book questions only; used for Median/P95 in summary.",
            "token_summary_scope": "Book Avg tokens use Book questions only; Overall Avg tokens include all 20 questions.",
        },
        "settings": {
            "collection_name": cfg.collection_name,
            "index_metadata_path": str(cfg.index_metadata_path),
            "embedding_model": cfg.embedding_model,
            "llm_model": cfg.llm_model,
            "chunk_size": cfg.chunk_size,
            "chunk_overlap": cfg.chunk_overlap,
            "top_k": cfg.top_k,
            "topic_routing_top_n": cfg.topic_routing_top_n,
            "benchmark_repeats": cfg.benchmark_repeats,
            "random_seed": cfg.random_seed,
        },
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_metadata_experiment(
    cfg: MetadataSettings = settings,
    question_ids: list[str] | None = None,
    selected_methods: list[str] | None = None,
    skip_benchmark: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

    benchmark_df = pd.DataFrame()
    benchmark_summary = pd.DataFrame()
    if not skip_benchmark:
        print("Running retrieval benchmark...")
        benchmark_df = run_retrieval_benchmark(cfg, vector_store, router, questions, methods)
        benchmark_summary = summarize_benchmark(benchmark_df)

    print("Running QA experiments...")
    detailed_df = run_qa_experiments(
        cfg,
        vector_store,
        router,
        llm,
        questions,
        methods,
        benchmark_summary,
    )
    scoring_df, mapping_df = create_scoring_sheet(detailed_df)
    return detailed_df, benchmark_df, scoring_df, mapping_df
