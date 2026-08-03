from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import _bootstrap  # noqa: F401

from acceptance_analysis import analyze_acceptance_report, format_confusion_table
from config import settings
from metadata_generator import generate_chunk_metadata
from prompts import load_allowed_topics
from src.llm_mlx import QwenMLX
from src.pdf_loader import chunk_pages, extract_pages


def load_acceptance_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_chunk_lookup() -> dict[str, object]:
    pages = extract_pages(settings.book_dir, settings.book_file)
    chunks = chunk_pages(
        pages,
        target_size=settings.chunk_target_size,
        max_size=settings.chunk_max_size,
        min_size=settings.chunk_min_size,
        overlap=settings.chunk_overlap,
    )
    return {chunk.chunk_id: chunk for chunk in chunks}


def text_preview(text: str, limit: int = 160) -> str:
    cleaned = text.replace("\n", " ").strip()
    return cleaned[:limit] + ("..." if len(cleaned) > limit else "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LLM-only metadata acceptance (generate only, no Qdrant write)."
    )
    parser.add_argument(
        "--set",
        choices=("dev", "holdout"),
        default="dev",
        help="Sample set to evaluate (default: dev calibration set).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: metadata_experiment/results/acceptance_{set}_{timestamp}.json)",
    )
    return parser.parse_args()


def resolve_manifest_path(sample_set: str) -> Path:
    if sample_set == "holdout":
        return settings.metadata_holdout_samples_file
    return settings.metadata_acceptance_samples_file


def main() -> None:
    args = parse_args()
    manifest_path = resolve_manifest_path(args.set)
    manifest = load_acceptance_manifest(manifest_path)
    if not manifest.get("samples"):
        raise RuntimeError(
            f"No samples in {manifest_path.name}. "
            "Populate the holdout set after prompt/ontology freeze."
        )
    chunk_lookup = build_chunk_lookup()
    topics = load_allowed_topics(settings.allowed_topics_file)
    allowed_ids = {topic.id for topic in topics}

    llm = QwenMLX(
        settings.llm_model,
        max_new_tokens=settings.metadata_max_new_tokens,
        temperature=settings.temperature,
    )
    print("Warming up metadata LLM...")
    llm.warm_up()

    rows = []
    missing_chunks: list[str] = []
    for sample in manifest["samples"]:
        chunk_id = sample["chunk_id"]
        chunk = chunk_lookup.get(chunk_id)
        if chunk is None:
            missing_chunks.append(chunk_id)
            continue
        print(f"[{sample.get('category_hint', '?')}] {chunk_id}")
        result = generate_chunk_metadata(
            llm,
            chunk.text,
            topics,
            max_retries=settings.metadata_gen_max_retries,
        )
        metadata = result.metadata
        rows.append(
            {
                "chunk_id": chunk_id,
                "category_hint": sample.get("category_hint", ""),
                "acceptable_topics": sample.get("acceptable_topics", []),
                "actual_topics": metadata.topics if metadata else [],
                "metadata_status": metadata.metadata_status if metadata else "failed",
                "characters": metadata.characters if metadata else [],
                "keywords": metadata.keywords if metadata else [],
                "text_preview": text_preview(chunk.text),
                "retries_used": result.retries_used,
                "json_parse_failure": result.json_parse_failure,
            }
        )
        if metadata and metadata.metadata_status == "ok":
            unknown = [topic_id for topic_id in metadata.topics if topic_id not in allowed_ids]
            if unknown:
                raise RuntimeError(f"Invalid topic ids for {chunk_id}: {unknown}")

    if missing_chunks:
        raise RuntimeError(f"Missing chunk ids in corpus: {sorted(set(missing_chunks))}")

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_path = args.output or (
        settings.results_dir / f"acceptance_{args.set}_{timestamp}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ontology_version = json.loads(settings.allowed_topics_file.read_text(encoding="utf-8")).get(
        "version", ""
    )
    manifest_ontology_version = manifest.get("ontology_version", "")
    if manifest_ontology_version and manifest_ontology_version != ontology_version:
        raise RuntimeError(
            f"Manifest ontology_version={manifest_ontology_version} "
            f"does not match allowed_topics version={ontology_version}"
        )
    payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "sample_set": manifest.get("set", args.set),
        "ontology_version": ontology_version,
        "manifest_version": manifest.get("version", ""),
        "manifest_path": str(manifest_path),
        "total_samples": len(rows),
        "samples": rows,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    analysis = analyze_acceptance_report(output_path)
    print(f"\nWrote acceptance report to {output_path}")
    print(format_confusion_table(analysis))
    print(
        f"\nPrimary pass: {analysis['primary_pass_count']}/{analysis['total_samples']} "
        f"({analysis['primary_pass_rate']:.1%})"
    )
    print(
        f"Topic-set pass: {analysis['topic_set_pass_count']}/{analysis['total_samples']} "
        f"({analysis['topic_set_pass_rate']:.1%})"
    )
    print(f"war as primary: {analysis['war_as_primary']}/{analysis['total_samples']}")


if __name__ == "__main__":
    main()
