from __future__ import annotations

from collections import defaultdict

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny

from config import MetadataSettings, settings
from metadata_retriever import MetadataVectorStore, build_router_from_settings
from topic_router import TopicRouter


def _count_filtered_chunks(client: QdrantClient, collection_name: str, topic_ids: list[str]) -> int:
    if not topic_ids:
        return 0
    query_filter = Filter(
        should=[FieldCondition(key="topics", match=MatchAny(any=topic_ids))]
    )
    count = 0
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=query_filter,
            limit=200,
            offset=offset,
            with_payload=False,
        )
        count += len(points)
        if offset is None:
            break
    return count


def analyze_book_filter_stats(cfg: MetadataSettings = settings) -> pd.DataFrame:
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

    rows = []
    for _, question_row in book_questions.iterrows():
        question_id = str(question_row["Question ID"])
        question = str(question_row["Question"])
        query_vector = vector_store.embed_query(question)
        predictions, _ = router.route(query_vector)
        top2_ids = [prediction.topic_id for prediction in predictions]

        per_topic_counts = {
            topic_id: _count_filtered_chunks(client, cfg.collection_name, [topic_id])
            for topic_id in top2_ids
        }
        or_count = _count_filtered_chunks(client, cfg.collection_name, top2_ids)
        rows.append(
            {
                "Question ID": question_id,
                "Question": question,
                "Top-1 Topic": top2_ids[0] if len(top2_ids) > 0 else "",
                "Top-2 Topic": top2_ids[1] if len(top2_ids) > 1 else "",
                "Top-1 Count": per_topic_counts.get(top2_ids[0], 0) if top2_ids else 0,
                "Top-2 Count": per_topic_counts.get(top2_ids[1], 0) if len(top2_ids) > 1 else 0,
                "Candidates Before": total_chunks,
                "Candidates After": or_count,
                "Reduction Rate": round(1 - or_count / total_chunks, 4) if total_chunks else 0.0,
            }
        )
    return pd.DataFrame(rows)
