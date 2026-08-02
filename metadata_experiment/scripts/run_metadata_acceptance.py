from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import _bootstrap  # noqa: F401

from config import settings
from metadata_generator import generate_chunk_metadata
from prompts import load_allowed_topics
from src.llm_mlx import QwenMLX
from src.pdf_loader import chunk_pages, extract_pages


def load_acceptance_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_chunk_lookup() -> dict[str, object]:
    pages = extract_pages(settings.book_dir, settings.book_file)
    chunks = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)
    return {chunk.chunk_id: chunk for chunk in chunks}


def text_preview(text: str, limit: int = 160) -> str:
    cleaned = text.replace("\n", " ").strip()
    return cleaned[:limit] + ("..." if len(cleaned) > limit else "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fixed-chunk metadata acceptance (generate only, no Qdrant write)."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: metadata_experiment/results/acceptance_{timestamp}.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_acceptance_manifest(settings.metadata_acceptance_samples_file)
    chunk_lookup = build_chunk_lookup()
    topics = load_allowed_topics(settings.allowed_topics_file)

    llm = QwenMLX(settings.llm_model, max_new_tokens=256, temperature=settings.temperature)
    print("Warming up metadata LLM...")
    llm.warm_up()

    rows = []
    missing_chunks: list[str] = []
    for category in manifest["categories"]:
        for chunk_id in category["chunk_ids"]:
            chunk = chunk_lookup.get(chunk_id)
            if chunk is None:
                missing_chunks.append(chunk_id)
                continue
            print(f"[{category['id']}] {chunk_id}")
            result = generate_chunk_metadata(
                llm,
                chunk.text,
                topics,
                max_retries=settings.metadata_gen_max_retries,
            )
            metadata = result.metadata
            rows.append(
                {
                    "category_id": category["id"],
                    "category_label": category["label"],
                    "chunk_id": chunk_id,
                    "expected_topics": category["expected_topics"],
                    "category_notes": category.get("notes", ""),
                    "actual_topics": metadata.topics if metadata else [],
                    "metadata_status": metadata.metadata_status if metadata else "failed",
                    "characters": metadata.characters if metadata else [],
                    "keywords": metadata.keywords if metadata else [],
                    "importance": metadata.importance if metadata else None,
                    "text_preview": text_preview(chunk.text),
                    "retries_used": result.retries_used,
                    "json_parse_failure": result.json_parse_failure,
                }
            )

    if missing_chunks:
        raise RuntimeError(f"Missing chunk ids in corpus: {sorted(set(missing_chunks))}")

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_path = args.output or (settings.results_dir / f"acceptance_{timestamp}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "topics_version": json.loads(settings.allowed_topics_file.read_text(encoding="utf-8")).get(
            "version", ""
        ),
        "manifest_version": manifest.get("version", ""),
        "total_samples": len(rows),
        "samples": rows,
        "review_instructions": (
            "Manually mark each row as pass/fail. A sample passes if actual_topics "
            "reasonably match the category (expected_topics are hints, not hard quotas)."
        ),
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\nWrote acceptance report to {output_path}")
    print(f"Samples: {len(rows)}")
    print("\nQuick preview:")
    for row in rows:
        print(
            f"  {row['chunk_id']} [{row['category_id']}] "
            f"expected={row['expected_topics']} actual={row['actual_topics']} "
            f"status={row['metadata_status']}"
        )


if __name__ == "__main__":
    main()
