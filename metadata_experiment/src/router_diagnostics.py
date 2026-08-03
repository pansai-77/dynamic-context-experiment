from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny

from config import MetadataSettings, settings
from metadata_retriever import MetadataVectorStore, build_router_from_settings


def load_book_gold_chunks(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    questions = payload.get("questions", {})
    if not questions:
        raise ValueError(f"No questions found in gold manifest: {path}")
    return questions


def first_gold_rank(retrieved_ids: list[str], gold_ids: list[str]) -> int | None:
    matched_ranks = [
        rank
        for rank, chunk_id in enumerate(retrieved_ids, start=1)
        if chunk_id in gold_ids
    ]
    return min(matched_ranks) if matched_ranks else None


def mrr_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int = 4) -> float:
    rank = first_gold_rank(retrieved_ids, gold_ids)
    if rank is None or rank > k:
        return 0.0
    return 1.0 / rank


def _chunk_topics_by_id(client: QdrantClient, collection_name: str) -> dict[str, list[str]]:
    topics_by_id: dict[str, list[str]] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            limit=200,
            offset=offset,
            with_payload=True,
        )
        for point in points:
            payload = point.payload or {}
            chunk_id = str(payload.get("chunk_id", point.id))
            topics_by_id[chunk_id] = list(payload.get("topics") or [])
        if offset is None:
            break
    return topics_by_id


def _filtered_chunk_ids(
    client: QdrantClient,
    collection_name: str,
    topic_ids: list[str],
) -> set[str]:
    if not topic_ids:
        return set()
    query_filter = Filter(
        should=[FieldCondition(key="topics", match=MatchAny(any=topic_ids))]
    )
    matched: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=query_filter,
            limit=200,
            offset=offset,
            with_payload=True,
        )
        for point in points:
            payload = point.payload or {}
            matched.add(str(payload.get("chunk_id", point.id)))
        if offset is None:
            break
    return matched


def analyze_router_diagnostics(cfg: MetadataSettings = settings) -> tuple[pd.DataFrame, dict]:
    gold_path = cfg.experiment_dir / "data" / "book_gold_chunks.json"
    gold_by_question = load_book_gold_chunks(gold_path)
    questions = pd.read_csv(cfg.questions_file)
    book_questions = questions[questions["Question Type"].str.lower() == "book"]

    vector_store = MetadataVectorStore(cfg.qdrant_path, cfg.collection_name, cfg.embedding_model)
    router = build_router_from_settings(
        vector_store,
        cfg.allowed_topics_file,
        cfg.topic_embeddings_file,
        cfg.topic_routing_top_n,
        cfg.embedding_model,
    )
    client = vector_store.client
    total_chunks = client.count(collection_name=cfg.collection_name, exact=True).count
    topics_by_id = _chunk_topics_by_id(client, cfg.collection_name)

    rows = []
    topic_top2_counts: dict[str, int] = {topic.id: 0 for topic in router.topics}

    for _, question_row in book_questions.iterrows():
        question_id = str(question_row["Question ID"])
        question = str(question_row["Question"])
        gold_entry = gold_by_question.get(question_id)
        if gold_entry is None:
            raise KeyError(f"Missing gold annotation for {question_id}")

        gold_ids = list(gold_entry["gold_chunk_ids"])
        gold_topics = sorted(
            {
                topic
                for chunk_id in gold_ids
                for topic in topics_by_id.get(chunk_id, [])
            }
        )

        query_vector = vector_store.embed_query(question)
        predictions, router_ms = router.route(query_vector)
        top2 = predictions[: cfg.topic_routing_top_n]
        top2_ids = [prediction.topic_id for prediction in top2]
        for topic_id in top2_ids:
            topic_top2_counts[topic_id] = topic_top2_counts.get(topic_id, 0) + 1

        score_gap = None
        if len(predictions) >= 2:
            score_gap = round(predictions[0].score - predictions[1].score, 4)

        filtered_ids = _filtered_chunk_ids(client, cfg.collection_name, top2_ids)
        gold_retained = any(chunk_id in filtered_ids for chunk_id in gold_ids)
        gold_topics_in_top2 = any(topic in top2_ids for topic in gold_topics)

        full_chunks, _ = vector_store.search_full(query_vector, top_k=cfg.top_k)
        meta_chunks, _, _ = vector_store.search_with_metadata(
            query_vector,
            router,
            top_k=cfg.top_k,
        )
        full_ids = [item.chunk.chunk_id for item in full_chunks]
        meta_ids = [item.chunk.chunk_id for item in meta_chunks]
        hit_a = any(chunk_id in full_ids for chunk_id in gold_ids)
        hit_b = any(chunk_id in meta_ids for chunk_id in gold_ids)

        rows.append(
            {
                "Question ID": question_id,
                "Question": question,
                "Gold Chunk IDs": ", ".join(gold_ids),
                "Gold Chunk Topics": ", ".join(gold_topics) if gold_topics else "(none)",
                "Top-1 Topic": top2_ids[0] if len(top2_ids) > 0 else "",
                "Top-1 Score": round(top2[0].score, 4) if len(top2) > 0 else None,
                "Top-2 Topic": top2_ids[1] if len(top2_ids) > 1 else "",
                "Top-2 Score": round(top2[1].score, 4) if len(top2) > 1 else None,
                "Score Gap": score_gap,
                "Gold Topic In Top-2": gold_topics_in_top2,
                "Candidates Before": total_chunks,
                "Candidates After": len(filtered_ids),
                "Gold Retained After Filter": gold_retained,
                "A Retrieved IDs": ", ".join(full_ids),
                "B Retrieved IDs": ", ".join(meta_ids),
                "A Hit@4": hit_a,
                "B Hit@4": hit_b,
                "A First Gold Rank": first_gold_rank(full_ids, gold_ids),
                "B First Gold Rank": first_gold_rank(meta_ids, gold_ids),
                "A MRR@4": round(mrr_at_k(full_ids, gold_ids, cfg.top_k), 4),
                "B MRR@4": round(mrr_at_k(meta_ids, gold_ids, cfg.top_k), 4),
                "Router Time (ms)": round(router_ms, 3),
            }
        )

    df = pd.DataFrame(rows)
    gold_retention_rate = float(df["Gold Retained After Filter"].mean())
    a_hit_rate = float(df["A Hit@4"].mean())
    b_hit_rate = float(df["B Hit@4"].mean())
    b_miss = df[~df["B Hit@4"]]
    b_miss_gold_filtered = b_miss[~b_miss["Gold Retained After Filter"]]
    b_miss_gold_retained_not_top4 = b_miss[b_miss["Gold Retained After Filter"]]

    summary = {
        "total_book_questions": len(df),
        "gold_retention_rate": round(gold_retention_rate, 4),
        "a_hit_at_4_rate": round(a_hit_rate, 4),
        "b_hit_at_4_rate": round(b_hit_rate, 4),
        "a_mrr_at_4": round(float(df["A MRR@4"].mean()), 4),
        "b_mrr_at_4": round(float(df["B MRR@4"].mean()), 4),
        "topic_top2_counts": topic_top2_counts,
        "gold_retained_count": int(df["Gold Retained After Filter"].sum()),
        "b_hit_without_gold_retained": int(
            df[~df["Gold Retained After Filter"] & df["B Hit@4"]].shape[0]
        ),
        "b_miss_count": int(b_miss.shape[0]),
        "b_miss_gold_filtered_count": int(b_miss_gold_filtered.shape[0]),
        "b_miss_gold_retained_not_top4_count": int(b_miss_gold_retained_not_top4.shape[0]),
        "b_miss_gold_filtered_question_ids": b_miss_gold_filtered["Question ID"].tolist(),
        "b_miss_gold_retained_not_top4_question_ids": b_miss_gold_retained_not_top4[
            "Question ID"
        ].tolist(),
        "b_miss_with_gold_retained": int(b_miss_gold_retained_not_top4.shape[0]),
    }
    return df, summary
