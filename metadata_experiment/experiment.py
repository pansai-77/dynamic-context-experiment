from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass
from typing import Any

import mlx.core as mx
import numpy as np
import pandas as pd

from src.llm_mlx import QwenMLX
from src.prompts import build_prompt

from .config import MetadataSettings
from .index_metadata import verify_chunk_parity_with_exp1, verify_index_metadata
from .logic import METHOD_B, METHODS, load_gold, should_retrieve
from .metrics import filter_accuracy, ranking_metrics
from .router import TopicRouter
from .vector_store import MetadataVectorStore


@dataclass
class MetadataExperimentRow:
    question_id: str
    question_type: str
    question: str
    method: str
    top_k: int
    used_retrieval: bool
    routed_topics: str
    candidates_before_filter: int | None
    candidates_after_filter: int | None
    router_time_ms: float
    vector_time_ms: float
    retrieval_time_ms: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    llm_time_ms: float
    total_time_ms: float
    answer: str
    retrieved_chunks: int
    retrieved_sources: str
    hit_at_4: float | None
    mrr_at_4: float | None
    filter_accuracy: float | None
    score_0_3: Any = None
    notes: str = ""


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)


def run_experiment(
    settings: MetadataSettings,
    limit: int | None = None,
    question_ids: list[str] | None = None,
) -> pd.DataFrame:
    _seed_everything(settings.random_seed)
    questions = pd.read_csv(settings.questions_file)
    if question_ids:
        missing = set(question_ids) - set(questions["Question ID"])
        if missing:
            raise ValueError(f"Unknown question IDs: {', '.join(sorted(missing))}")
        order = {qid: index for index, qid in enumerate(question_ids)}
        questions = questions[questions["Question ID"].isin(question_ids)].copy()
        questions["_order"] = questions["Question ID"].map(order)
        questions = questions.sort_values("_order").drop(columns="_order")
    if limit:
        questions = questions.head(limit)

    index_manifest = verify_index_metadata(settings)
    parity_issues = verify_chunk_parity_with_exp1(settings)
    if parity_issues:
        raise ValueError(
            "Metadata index chunk set does not match experiment 1 index:\n- "
            + "\n- ".join(parity_issues)
            + "\nRebuild both indexes with the same PDF and chunk settings."
        )
    print(
        "Index metadata:",
        f"embedding={index_manifest.embedding_model},",
        f"chunk_strategy={index_manifest.chunk_strategy},",
        f"chunks={len(index_manifest.chunk_ids)},",
        f"sources={index_manifest.source_files}",
    )

    gold = load_gold(settings.gold_file)
    store = MetadataVectorStore(
        settings.qdrant_path, settings.collection_name, settings.embedding_model
    )
    total_candidates = store.total_candidates()
    router = TopicRouter(store.embedding_model)
    # Warm-up outside timed measurements.
    store.search("预热检索", top_k=1)
    router.route("预热路由", settings.router_top_n)
    llm = QwenMLX(settings.llm_model, settings.max_new_tokens, settings.temperature)
    llm.warm_up()

    rows: list[MetadataExperimentRow] = []
    for _, question_row in questions.iterrows():
        qid = str(question_row["Question ID"])
        qtype = str(question_row["Question Type"])
        question = str(question_row["Question"])
        use_retrieval = should_retrieve(qtype)
        annotation = gold.get(qid, {"topics": [], "chunks": []})

        for method in METHODS:
            started = time.perf_counter()
            routed_topics: list[str] = []
            router_ms = 0.0
            vector_ms = 0.0
            before: int | None = None
            after: int | None = None
            retrieved = []

            if use_retrieval:
                before = total_candidates
                if method == METHOD_B:
                    routed_topics, router_ms = router.route(question, settings.router_top_n)
                    after = store.candidate_count(routed_topics)
                    retrieved, vector_ms = store.search(
                        question, settings.retrieval_top_k, routed_topics
                    )
                else:
                    after = total_candidates
                    retrieved, vector_ms = store.search(question, settings.retrieval_top_k)

            generation = llm.answer(build_prompt(question, qtype, retrieved))
            total_ms = (time.perf_counter() - started) * 1000
            retrieved_ids = [item.chunk.chunk_id for item in retrieved]
            hit, mrr = ranking_metrics(retrieved_ids, annotation["chunks"])
            accuracy = (
                filter_accuracy(routed_topics, annotation["topics"])
                if method == METHOD_B and use_retrieval
                else None
            )
            rows.append(MetadataExperimentRow(
                question_id=qid,
                question_type=qtype,
                question=question,
                method=method,
                top_k=settings.retrieval_top_k if use_retrieval else 0,
                used_retrieval=use_retrieval,
                routed_topics=" | ".join(routed_topics),
                candidates_before_filter=before,
                candidates_after_filter=after,
                router_time_ms=round(router_ms, 3),
                vector_time_ms=round(vector_ms, 3),
                retrieval_time_ms=round(router_ms + vector_ms, 3),
                input_tokens=generation.input_tokens,
                output_tokens=generation.output_tokens,
                total_tokens=generation.total_tokens,
                llm_time_ms=round(generation.llm_time_ms, 3),
                total_time_ms=round(total_ms, 3),
                answer=generation.answer,
                retrieved_chunks=len(retrieved),
                retrieved_sources="; ".join(
                    f"{item.chunk.chunk_id}:p{item.chunk.page_start}"
                    + (f"-{item.chunk.page_end}" if item.chunk.page_end != item.chunk.page_start else "")
                    + f":{item.score:.3f}"
                    for item in retrieved
                ),
                hit_at_4=hit,
                mrr_at_4=mrr,
                filter_accuracy=accuracy,
            ))
    return pd.DataFrame(asdict(row) for row in rows)

