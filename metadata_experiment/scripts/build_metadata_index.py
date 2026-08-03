import json
from dataclasses import asdict

import _bootstrap  # noqa: F401

from config import settings
from index_metadata import expected_metadata, write_index_metadata
from metadata_generator import generate_chunk_metadata
from metadata_retriever import MetadataVectorStore, build_topic_embeddings
from prompts import load_allowed_topics
from router_diagnostics import load_book_gold_chunks
from src.llm_mlx import QwenMLX
from src.pdf_loader import chunk_pages, extract_pages, resolve_pdf_files
from topic_coverage import build_topic_coverage_report, topic_coverage_warnings

def main() -> None:
    pdf_files = resolve_pdf_files(settings.book_dir, settings.book_file)
    pages = extract_pages(settings.book_dir, settings.book_file)
    chunks = chunk_pages(
        pages,
        target_size=settings.chunk_target_size,
        max_size=settings.chunk_max_size,
        min_size=settings.chunk_min_size,
        overlap=settings.chunk_overlap,
    )
    topics = load_allowed_topics(settings.allowed_topics_file)
    allowed_ids = {topic.id for topic in topics}

    print(f"Extracted {len(pages)} pages; created {len(chunks)} chunks.")
    llm = QwenMLX(
        settings.llm_model,
        max_new_tokens=settings.metadata_max_new_tokens,
        temperature=settings.temperature,
    )
    print("Warming up metadata LLM...")
    llm.warm_up()

    metadata_by_chunk_id = {}
    failed_chunk_ids: list[str] = []
    json_parse_failures = 0
    invalid_topic_count = 0
    retry_count = 0
    metadata_ok_count = 0

    for index, chunk in enumerate(chunks, start=1):
        print(f"[{index}/{len(chunks)}] metadata for {chunk.chunk_id}")
        result = generate_chunk_metadata(
            llm,
            chunk.text,
            topics,
            max_retries=settings.metadata_gen_max_retries,
        )
        if result.json_parse_failure:
            json_parse_failures += 1
        if result.invalid_topic_ids:
            invalid_topic_count += len(result.invalid_topic_ids)
        retry_count += result.retries_used
        metadata = result.metadata
        if metadata is None:
            raise RuntimeError(f"Metadata generation returned None for {chunk.chunk_id}")
        metadata_by_chunk_id[chunk.chunk_id] = metadata
        if metadata.metadata_status == "ok":
            metadata_ok_count += 1
            unknown = [topic_id for topic_id in metadata.topics if topic_id not in allowed_ids]
            if unknown:
                raise RuntimeError(f"Invalid topic ids stored for {chunk.chunk_id}: {unknown}")
        else:
            failed_chunk_ids.append(chunk.chunk_id)

    gold_manifest = settings.experiment_dir / "data" / "book_gold_chunks.json"
    if gold_manifest.exists():
        gold_by_question = load_book_gold_chunks(gold_manifest)
        gold_ids = {
            chunk_id
            for entry in gold_by_question.values()
            for chunk_id in entry["gold_chunk_ids"]
        }
        gold_failures = []
        for chunk_id in sorted(gold_ids):
            metadata = metadata_by_chunk_id.get(chunk_id)
            if metadata is None:
                gold_failures.append(f"{chunk_id}: missing metadata")
            elif metadata.metadata_status != "ok":
                gold_failures.append(f"{chunk_id}: metadata_status=failed")
            elif not metadata.topics:
                gold_failures.append(f"{chunk_id}: empty topics")
        if gold_failures:
            raise RuntimeError(
                "Gold chunk metadata check failed before indexing:\n- "
                + "\n- ".join(gold_failures)
            )
        print(f"Gold chunk metadata check passed ({len(gold_ids)} chunks).")

    store = MetadataVectorStore(
        settings.qdrant_path,
        settings.collection_name,
        settings.embedding_model,
    )
    print("Building metadata index (text-only embeddings)...")
    store.rebuild(chunks, metadata_by_chunk_id)
    build_topic_embeddings(
        store,
        topics,
        settings.topic_embeddings_file,
        settings.embedding_model,
        settings.allowed_topics_file,
    )

    metadata = expected_metadata(settings, [path.name for path in pdf_files])
    write_index_metadata(settings.index_metadata_path, metadata)
    print(f"Wrote index metadata to {settings.index_metadata_path}")

    report = {
        "total_chunks": len(chunks),
        "indexed_chunks": len(chunks),
        "metadata_ok_count": metadata_ok_count,
        "metadata_failed_count": len(chunks) - metadata_ok_count,
        "metadata_coverage": round(metadata_ok_count / len(chunks), 4),
        "json_parse_failure_count": json_parse_failures,
        "invalid_topic_count": invalid_topic_count,
        "retry_count": retry_count,
        "chunk_embedding_policy": "text_only",
        "failed_chunk_ids": failed_chunk_ids,
        "max_topics_per_chunk": 2,
    }
    settings.index_build_report_file.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote index build report to {settings.index_build_report_file}")
    print(json.dumps(report, indent=2, ensure_ascii=False))

    coverage_report = build_topic_coverage_report(metadata_by_chunk_id, allowed_ids)
    settings.topic_coverage_report_file.write_text(
        json.dumps(coverage_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote topic coverage report to {settings.topic_coverage_report_file}")
    print(json.dumps(coverage_report, indent=2, ensure_ascii=False))
    for warning in topic_coverage_warnings(coverage_report):
        print(warning)
    print("Metadata index built successfully.")

if __name__ == "__main__":
    main()
