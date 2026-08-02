# Experiment 2: LLM-generated Metadata RAG (Phase 1)

## Research Objective

Investigate whether metadata generated during the indexing stage can improve RAG retrieval efficiency while maintaining comparable answer quality.

## Hypotheses

- **H1:** Metadata filtering is expected to reduce retrieval latency.
- **H2:** Metadata filtering is expected to reduce input tokens.
- **H3:** Metadata filtering maintains comparable answer quality.

## Methods

| Method | Description |
|--------|-------------|
| **A: Query-Aware Top-4** | Book → full-corpus Top-4 retrieval |
| **B: Query-Aware + Metadata Top-4** | Book → Top-2 topic routing → OR metadata filter → Top-4 |

General / Rewrite questions skip retrieval in both arms.

**Formal main experiment:** fixed Top-2 → OR → Top-4, **no fallback**.

**Supplementary only:** `--with-fallback` expands Top-2 to Top-3/4 when the filter returns 0 chunks.

## Primary Metrics

Guide-aligned metrics only:

| Metric | Detailed | Summary |
|--------|----------|---------|
| Retrieval / LLM / Total Time | per row | averaged |
| Input / Output / Total Tokens | per row | Book-only average |
| Retrieved Chunks | per row (0 for General/Rewrite) | — |
| Answer quality | `Score(0-3)` per row | Book Score, Overall Score |

Phase 1 does **not** use gold chunks, Hit@4, or MRR.

## Setup

Uses the same `.env` and `requirements.txt` as Experiment 1 at the repository root.

```bash
cp .env.example .env
pip install -r requirements.txt
```

## Metadata v3 (Current)

- **Ontology:** 7 event-level topics — `war`, `politics`, `gambling`, `family`, `medical`, `labor`, `livelihood`
- **Prompt:** short rules + few-shot examples; `topics` 0–1; `keywords` ×3; no `importance`
- **Pilot v2.1 index:** diagnostic only; do not use for formal H1–H3

## Recommended Execution Order

1. Keep **Pilot v2.1** index and reports (diagnostic only).
2. Run **v3 acceptance** on the 40-chunk development set.
3. Analyze pass rate + confusion matrix; target **≥70% pass** before full rebuild.
4. Full rebuild, then coverage + filter stats + Top-4 spot-check.
5. Run formal A/B (no `--with-fallback`).

```text
Pilot v2.1 (keep)
→ run_metadata_acceptance.py
→ analyze_acceptance.py
→ build_metadata_index.py
→ topic_coverage_report.json
→ analyze_filter_stats.py
→ Top-4 spot-check
→ run_metadata_experiment.py
```

## Commands

v3 metadata acceptance (generate only, no Qdrant write):

```bash
python metadata_experiment/scripts/run_metadata_acceptance.py
python metadata_experiment/scripts/analyze_acceptance.py
```

Build the metadata index (offline LLM metadata + text-only embeddings):

```bash
python metadata_experiment/scripts/build_metadata_index.py
```

Inspect topic coverage (written during index build):

```bash
cat metadata_experiment/topic_coverage_report.json
```

Analyze Book-question filter reduction stats:

```bash
python metadata_experiment/scripts/analyze_filter_stats.py
python metadata_experiment/scripts/analyze_filter_stats.py --output metadata_experiment/filter_stats.json
```

Run Phase 1 experiment (formal):

```bash
python metadata_experiment/scripts/run_metadata_experiment.py
```

Supplementary run with topic expansion fallback:

```bash
python metadata_experiment/scripts/run_metadata_experiment.py --with-fallback
```

Fill `Score(0-3)` in `detailed_results.xlsx`, then regenerate summary:

```bash
python metadata_experiment/scripts/summarise_results.py --run-dir 20260802_120000
```

Run tests:

```bash
pytest metadata_experiment/tests/
```

## Metadata Semantics

- `topics=[]` with `metadata_status="ok"`: no sufficiently clear topic for this chunk.
- `metadata_status="failed"`: JSON parse / field validation failure.
- Acceptance uses `acceptable_topics` (multiple valid label sets), not strict single gold labels.
- Config: `metadata_experiment/data/allowed_topics.json` (v3.0)
- Manifest: `metadata_experiment/data/metadata_acceptance_samples.json` (v3.0)

## Outputs

Index build:

- `metadata_experiment/index_build_report.json`
- `metadata_experiment/topic_coverage_report.json`

Results are written to `metadata_experiment/results/{timestamp}/`:

- `detailed_results.xlsx`
- `summary_results.xlsx`
- `run_config.json`

## Isolation

- Index metadata: `metadata_experiment/index_metadata.json`
- Qdrant collection: `huozhe_meta`
- Experiment 1 remains in the repository root and is not modified.
