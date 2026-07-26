# Dynamic Context Experiment

Local feasibility experiment measuring how RAG context strategies affect token usage, latency, estimated cost and answer quality.

## Methods
1. Baseline RAG: Top-8
2. Standard RAG: Top-4
3. Minimal RAG: Top-2
4. No RAG
5. Query-Aware: Book → Top-4, General/Rewrite → No RAG
6. Query-Aware + Top-2: optional

Knowledge base: first 143 pages of *An Introduction to Statistical Learning with Applications in Python* (Chapters 1–3).

## Stack
- Apple Silicon local inference
- Qwen2.5-3B-Instruct 4-bit through MLX
- BAAI/bge-small-en-v1.5
- Qdrant local mode
- PyMuPDF
- Excel output

## Setup
```bash
cd dynamic-context-experiment
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

The first run downloads the embedding and Qwen models from Hugging Face.

## Run
```bash
python scripts/build_index.py
python scripts/run_experiments.py --limit 3
python scripts/run_experiments.py
python scripts/run_experiments.py --include-optional
```

Default runs use the five core methods. Add `--include-optional` to also run Query-Aware + Top-2.

Each run creates a timestamped directory under `results/`, for example `results/20260726_093512/`, containing:
- `detailed_results.xlsx`
- `summary_results.xlsx`
- `run_config.json`

## Manual scoring
Score every answer in the run directory's `detailed_results.xlsx`, column `Score(0-3)`:
- 3 correct and complete
- 2 mostly correct
- 1 partially correct
- 0 incorrect

Then regenerate summaries for the latest run:
```bash
python scripts/summarise_results.py
```

Or target a specific run:
```bash
python scripts/summarise_results.py --run-dir 20260726_093512
```

## Experimental control
Keep the LLM, embedding model, chunking parameters, questions and generation parameters constant. Warm up the model before timing. Compare Book, General and Rewrite questions separately. Token and latency reductions are calculated against Top-8 Baseline.
