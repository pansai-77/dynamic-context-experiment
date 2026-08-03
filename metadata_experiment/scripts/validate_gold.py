from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from metadata_experiment.config import settings
from metadata_experiment.index_metadata import manifest_path, read_index_manifest
from metadata_experiment.logic import load_gold


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate gold annotations before experiment 2.")
    parser.add_argument(
        "--require-chunk-ids",
        action="store_true",
        help="Fail when any Book question is missing Gold Chunk IDs.",
    )
    args = parser.parse_args()

    manifest = read_index_manifest(manifest_path(settings.qdrant_path))
    if manifest is None:
        raise SystemExit(
            f"Missing metadata index manifest at {manifest_path(settings.qdrant_path)}. "
            "Run metadata_experiment/scripts/build_index.py first."
        )

    known_chunk_ids = set(manifest.chunk_ids)
    gold = load_gold(settings.gold_file)
    missing_topics: list[str] = []
    missing_chunks: list[str] = []
    unknown_chunks: list[str] = []

    for question_id, annotation in sorted(gold.items()):
        if not annotation["topics"]:
            missing_topics.append(question_id)
        if not annotation["chunks"]:
            missing_chunks.append(question_id)
            continue
        for chunk_id in annotation["chunks"]:
            if chunk_id not in known_chunk_ids:
                unknown_chunks.append(f"{question_id}:{chunk_id}")

    print(f"Loaded {len(gold)} gold rows from {settings.gold_file}")
    print(f"Index catalog contains {len(known_chunk_ids)} chunk ids.")

    if missing_topics:
        print(f"Warning: missing Gold Topics for {', '.join(missing_topics)}")
    if missing_chunks:
        print(f"Warning: missing Gold Chunk IDs for {', '.join(missing_chunks)}")
    if unknown_chunks:
        print("Error: unknown chunk ids referenced in gold annotations:")
        for item in unknown_chunks:
            print(f"  - {item}")
        raise SystemExit(1)

    if args.require_chunk_ids and missing_chunks:
        raise SystemExit(
            "Gold Chunk IDs are required before formal experiment 2 runs. "
            "Fill metadata_experiment/data/gold_annotations.csv using data/index_catalog.xlsx."
        )

    print("Gold annotation validation passed.")


if __name__ == "__main__":
    main()
