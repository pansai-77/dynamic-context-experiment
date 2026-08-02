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

## Primary Metrics

Retrieval Time, LLM Time, Total Time, Input/Output/Total Tokens, Book Score, Overall Score.

Phase 1 does **not** use gold chunks, Hit@4, or MRR.

## Setup

Uses the same `.env` and `requirements.txt` as Experiment 1 at the repository root.

```bash
cp .env.example .env
pip install -r requirements.txt
```

## Commands

Build the metadata index (offline LLM metadata + text-only embeddings):

```bash
python metadata_experiment/scripts/build_metadata_index.py
```

Run Phase 1 experiment:

```bash
python metadata_experiment/scripts/run_metadata_experiment.py
```

After blind scoring `scoring_sheet.xlsx`, merge scores:

```bash
python metadata_experiment/scripts/summarise_results.py --run-dir 20260802_120000
```

Run tests:

```bash
pytest metadata_experiment/tests/
```

## Outputs

Results are written to `metadata_experiment/results/{timestamp}/`:

- `detailed_results.xlsx`
- `scoring_sheet.xlsx` (blind scoring)
- `scoring_mapping.csv`
- `retrieval_benchmark.xlsx`
- `summary_results.xlsx`
- `run_config.json`

## Isolation

- Index metadata: `metadata_experiment/index_metadata.json`
- Qdrant collection: `huozhe_meta`
- Experiment 1 remains in the repository root and is not modified.
