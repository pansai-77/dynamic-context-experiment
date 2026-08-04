from __future__ import annotations

import csv
import json
import random
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import numpy as np
import pandas as pd

from src.models import Chunk

from .classification import (
    ChunkClassificationError,
    ChunkClassificationResult,
    ChunkTopicClassifier,
    compute_cache_version,
    create_topic_classifier,
)
from .classification_prompts import CLASSIFICATION_PROMPT_VERSION
from .config import MetadataSettings
from .index_metadata import (
    expected_manifest,
    manifest_path,
    verify_chunk_parity_with_exp1,
    write_index_manifest,
)
from .metadata_quality import evaluate_distribution_quality, format_distribution_report
from .topics import annotate_auxiliary_metadata, topic_names
from .retrieval import MetadataVectorStore

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)


def _cell_value(cell: ET.Element) -> str:
    inline = cell.find("m:is", NS)
    if inline is not None:
        return "".join((t.text or "") for t in inline.findall(".//m:t", NS))
    value = cell.find("m:v", NS)
    return (value.text or "") if value is not None else ""


def load_chunks_from_catalog(catalog_path: Path) -> list[Chunk]:
    with zipfile.ZipFile(catalog_path) as archive:
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        rows = [
            [_cell_value(cell) for cell in row.findall("m:c", NS)]
            for row in sheet.findall("m:sheetData/m:row", NS)
        ]

    header = rows[0]
    index = {name: position for position, name in enumerate(header)}
    chunks: list[Chunk] = []
    for chunk_index, row in enumerate(rows[1:], start=1):
        chunk_id = row[index["Chunk ID"]]
        page_start = int(float(row[index["Page Start"]]))
        page_end = int(float(row[index["Page End"]]))
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                text=row[index["Text"]],
                source_file="index_catalog.xlsx",
                page_number=page_start,
                page_start=page_start,
                page_end=page_end,
                chunk_index=chunk_index,
            )
        )
    return chunks


def export_original_metadata(
    path: Path,
    *,
    topics_by_chunk_id: dict[str, list[str]],
    raw_outputs: dict[str, str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["chunk_id", "raw_output", "parsed_topics", "final_topics"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for chunk_id in sorted(topics_by_chunk_id):
            topics = topics_by_chunk_id[chunk_id]
            writer.writerow({
                "chunk_id": chunk_id,
                "raw_output": (raw_outputs or {}).get(chunk_id, ""),
                "parsed_topics": " | ".join(topics),
                "final_topics": " | ".join(topics),
            })


def build_chunk_payloads(
    chunks: list[Chunk],
    topics_by_chunk_id: dict[str, list[str]],
) -> dict[str, dict]:
    payloads: dict[str, dict] = {}
    for chunk in chunks:
        auxiliary = annotate_auxiliary_metadata(chunk)
        topics = topics_by_chunk_id[chunk.chunk_id]
        payloads[chunk.chunk_id] = {
            **auxiliary,
            "topics": topics,
        }
    return payloads


def classify_chunks(
    chunks: list[Chunk],
    classifier: ChunkTopicClassifier,
    *,
    show_details: bool = False,
    use_cache: bool = True,
) -> list[ChunkClassificationResult]:
    results: list[ChunkClassificationResult] = []
    total = len(chunks)

    for index, chunk in enumerate(chunks, start=1):
        print(f"[{index}/{total}] Classifying {chunk.chunk_id}...", flush=True)
        try:
            result = classifier.classify(chunk, use_cache=use_cache)
        except ChunkClassificationError as exc:
            if exc.last_response:
                print(f"  Last raw response: {exc.last_response}")
            raise

        results.append(result)
        if show_details:
            preview = chunk.text.replace("\n", " ")
            if len(preview) > 120:
                preview = preview[:120] + "..."
            print(f"  Text: {preview}")
            print(f"  Raw: {result.raw_response}")
            print(f"  Parsed: {result.parsed_topics}")
            print(f"  Warnings: {list(result.validation_warnings) or '-'}")
            print(f"  Final: {result.final_topics}")

    return results


def topics_from_results(results: list[ChunkClassificationResult]) -> dict[str, list[str]]:
    return {result.chunk_id: list(result.final_topics) for result in results}


def print_pre_build_quality_report(
    results: list[ChunkClassificationResult],
    settings: MetadataSettings,
    *,
    indexed_topics: dict[str, list[str]],
) -> None:
    cache_version = compute_cache_version(
        llm_model=settings.llm_model,
        temperature=settings.temperature,
        max_new_tokens=settings.classification_max_new_tokens,
    )
    distribution = evaluate_distribution_quality(indexed_topics)
    retry_records = [
        {
            "chunk_id": result.chunk_id,
            "attempts": result.attempts,
            "structure_errors": list(result.structure_errors),
        }
        for result in results
        if result.attempts > 1 or result.structure_errors
    ]
    warning_records = [
        {
            "chunk_id": result.chunk_id,
            "warnings": list(result.validation_warnings),
        }
        for result in results
        if result.validation_warnings
    ]

    print()
    print("=" * 80)
    print("Metadata pre-build quality report")
    print("=" * 80)
    print(f"Prompt version: {CLASSIFICATION_PROMPT_VERSION}")
    print(f"Cache version: {cache_version}")
    print(f"Classified chunks: {len(results)}")
    print(f"Structure retry records: {len(retry_records)}")
    print(f"Content audit warnings: {len(warning_records)}")
    print()
    print(format_distribution_report(distribution))
    print()
    print("Per-chunk classification records:")
    for result in results:
        print(f"  {result.chunk_id}:")
        print(f"    raw_output: {result.raw_response!r}")
        print(f"    parsed_topics: {result.parsed_topics}")
        print(f"    validation_warnings: {list(result.validation_warnings) or '-'}")
        print(f"    final_topics: {result.final_topics}")

    if retry_records:
        print()
        print("Structure retry records:")
        for record in retry_records:
            print(f"  - {record}")

    if warning_records:
        print()
        print("Content audit warnings (informational only):")
        for record in warning_records:
            print(f"  - {record}")

    if distribution.warnings:
        print()
        print("Distribution warnings:")
        for warning in distribution.warnings:
            print(f"  - {warning}")


def run_full_index_build(
    settings: MetadataSettings,
    chunks: list[Chunk],
    topics_by_chunk_id: dict[str, list[str]],
    catalog_path: Path,
    *,
    skip_parity_check: bool = False,
) -> None:
    payloads = build_chunk_payloads(chunks, topics_by_chunk_id)

    print(f"Loading embedding model ({settings.embedding_model}) and writing Qdrant index...")
    store = MetadataVectorStore(
        settings.qdrant_path,
        settings.collection_name,
        settings.embedding_model,
    )
    try:
        store.rebuild(chunks, payloads)
    finally:
        store.close()

    catalog_rows = []
    for chunk in chunks:
        payload = payloads[chunk.chunk_id]
        catalog_rows.append({
            "Chunk ID": chunk.chunk_id,
            "Page Start": chunk.page_start,
            "Page End": chunk.page_end,
            "Topics": " | ".join(payload["topics"]),
            "Characters": " | ".join(payload["characters"]),
            "Keywords": " | ".join(payload["keywords"]),
            "Importance": payload["importance"],
            "Text": chunk.text,
        })

    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(catalog_rows).to_excel(catalog_path, index=False, sheet_name="Index Catalog")

    manifest = expected_manifest(settings, chunks, topics=topic_names())
    output_manifest_path = manifest_path(settings.qdrant_path)
    write_index_manifest(output_manifest_path, manifest)

    topic_counts = Counter()
    for topics in topics_by_chunk_id.values():
        topic_counts.update(topics)

    print()
    print(f"Total chunks: {len(chunks)}")
    print(f"Successfully indexed: {len(topics_by_chunk_id)}")
    print("Topic distribution:")
    for topic, count in sorted(topic_counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {topic}: {count}")
    print(f"Collection: {settings.collection_name}")
    print(f"Manifest: {output_manifest_path}")
    print(f"Index catalog: {catalog_path}")

    meta_chunk_ids = sorted(chunk.chunk_id for chunk in chunks)
    if skip_parity_check:
        print("Skipped experiment 1 chunk parity check for partial index build.")
        return

    print("Checking chunk parity with experiment 1 index...")
    parity_issues = verify_chunk_parity_with_exp1(settings, meta_chunk_ids=meta_chunk_ids)
    if parity_issues:
        print("Warning: metadata index differs from experiment 1 index:")
        for issue in parity_issues:
            print(f"  - {issue}")
        return

    print("Experiment 1 chunk parity check passed.")


def export_classification_audit(
    results: list[ChunkClassificationResult],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(result) for result in results]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
