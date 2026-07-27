# Dynamic Context Experiment / 动态 Context 实验

**English:** A reproducible local RAG experiment framework for comparing context retrieval strategies on token usage, latency, estimated cost, and answer quality.

**中文：** 一个可复现的本地 RAG 实验框架，用于比较不同 Context 检索策略对 Token 用量、延迟、估算成本与回答质量的影响。

---

## 1. Research Goal / 研究目标

**English**

This is an academic feasibility study, not a production system. The project keeps the LLM, embedding model, chunking settings, question set, and generation parameters fixed, and changes **only the context retrieval strategy** between runs.

**Metrics tracked**

- Input / Output / Total Tokens
- Retrieval Time / LLM Time / Total Time
- Output Tokens/sec
- Estimated Cost (USD)
- Retrieved Chunks
- Answer + manual Accuracy Score (0–3)

**中文**

本项目属于学术研究用的可行性实验，不是生产系统。实验过程中固定 LLM、Embedding、切块参数、题目集与生成参数，**每次只改变 Context 检索策略**。

**记录的指标**

- Input / Output / Total Tokens
- Retrieval Time / LLM Time / Total Time
- Output Tokens/sec
- Estimated Cost (USD)
- Retrieved Chunks
- Answer + 人工 Accuracy Score（0–3）

---

## 2. Experiment Methods / 实验方法

| # | Method | Top-k | Retrieval rule |
|---|--------|-------|----------------|
| 1 | Baseline RAG | 8 | Always retrieve |
| 2 | Standard RAG | 4 | Always retrieve |
| 3 | Minimal RAG | 2 | Always retrieve |
| 4 | No RAG | 0 | Never retrieve |
| 5 | Query-Aware | 4 for Book | Book → retrieve; General / Rewrite → no retrieval |
| 6 | Query-Aware + Top-2 *(optional)* | 2 for Book | Same as Query-Aware, but Top-2 for Book questions |

**Default runs:** methods 1–5 (100 rows = 20 questions × 5 methods)

**With `--include-optional`:** methods 1–6 (120 rows)

| # | 方法 | Top-k | 检索规则 |
|---|------|-------|----------|
| 1 | Baseline RAG | 8 | 始终检索 |
| 2 | Standard RAG | 4 | 始终检索 |
| 3 | Minimal RAG | 2 | 始终检索 |
| 4 | No RAG | 0 | 不检索 |
| 5 | Query-Aware | Book 用 4 | Book 检索；General / Rewrite 不检索 |
| 6 | Query-Aware + Top-2 *（可选）* | Book 用 2 | 同 Query-Aware，但 Book 题只用 Top-2 |

**默认运行：** 前 5 种方法（100 条 = 20 题 × 5 方法）

**加 `--include-optional`：** 6 种方法（120 条）

---

## 3. Dataset / 数据集

**English**

| Item | Detail |
|------|--------|
| Knowledge base | *An Introduction to Statistical Learning with Applications in Python* (ISLP) |
| Pages used | First **143 PDF pages** (Chapters 1–3: Introduction, Statistical Learning, Linear Regression) |
| PDF path | `data/book/ISLR_Python_Pages_1_to_143.pdf` |
| Questions | `data/questions/questions.csv` — 20 questions total |
| Question types | 10 Book + 5 General + 5 Rewrite |
| Book questions | Must be answerable from the indexed book content |
| General / Rewrite | Do not require book retrieval (Query-Aware skips retrieval for these types) |

**中文**

| 项目 | 说明 |
|------|------|
| 知识库 | 《An Introduction to Statistical Learning with Applications in Python》(ISLP) |
| 使用页数 | 前 **143 页 PDF**（第 1–3 章：Introduction、Statistical Learning、Linear Regression） |
| PDF 路径 | `data/book/ISLR_Python_Pages_1_to_143.pdf` |
| 问题集 | `data/questions/questions.csv`，共 20 题 |
| 题型 | 10 Book + 5 General + 5 Rewrite |
| Book 题 | 必须能依据书内内容作答 |
| General / Rewrite | 不依赖书本检索（Query-Aware 对这两类不检索） |

---

## 4. Tech Stack / 技术栈

| Component | Choice |
|-----------|--------|
| Platform | Apple Silicon (local) |
| LLM | Qwen2.5-3B-Instruct 4-bit via **mlx-lm** |
| Embedding | **BAAI/bge-small-en-v1.5** (fixed for all runs) |
| Vector DB | **Qdrant local mode** (`qdrant_storage/`) |
| PDF parsing | PyMuPDF |
| Python | **3.11** |
| Output | Excel + JSON metadata |

| 组件 | 选型 |
|------|------|
| 运行环境 | Apple Silicon 本地 |
| LLM | Qwen2.5-3B-Instruct 4-bit，通过 **mlx-lm** |
| Embedding | **BAAI/bge-small-en-v1.5**（全程固定） |
| 向量库 | **Qdrant 本地模式**（`qdrant_storage/`） |
| PDF 解析 | PyMuPDF |
| Python | **3.11** |
| 输出 | Excel + JSON 元数据 |

---

## 5. Project Structure / 项目结构

```
dynamic-context-experiment/
├── data/
│   ├── book/                    # Knowledge-base PDF(s)
│   └── questions/questions.csv  # Fixed 20-question benchmark
├── scripts/
│   ├── build_index.py           # PDF → chunks → Qdrant (run once after book change)
│   ├── run_experiments.py       # Main experiment runner
│   └── summarise_results.py     # Regenerate summary after manual scoring
├── src/
│   ├── config.py                # Settings from .env
│   ├── pdf_loader.py            # Extract & chunk PDF pages
│   ├── vector_store.py          # Embedding + Qdrant search
│   ├── prompts.py               # Prompt templates
│   ├── llm_mlx.py               # Local Qwen inference
│   ├── experiment.py            # Core experiment loop & methods
│   ├── reporting.py             # Summary workbooks
│   └── run_metadata.py          # Timestamped run dirs & run_config.json
├── qdrant_storage/              # Local vector index (generated)
├── results/                     # Timestamped experiment outputs
├── .env.example                 # Config template
└── requirements.txt
```

**中文对照**

- `build_index.py`：换书或改 chunk 参数后执行，构建向量索引
- `run_experiments.py`：运行实验主入口
- `summarise_results.py`：人工打分后重新生成汇总
- `experiment.py`：六种方法定义与指标采集核心逻辑

---

## 6. Setup / 环境配置

```bash
cd dynamic-context-experiment
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

**English:** The first run downloads the embedding and LLM weights from Hugging Face. Ensure network access is available.

**中文：** 首次运行会从 Hugging Face 下载 Embedding 与 LLM 模型，请确保网络可用。

### Key `.env` parameters / 主要配置项

| Variable | Default | Meaning |
|----------|---------|---------|
| `LLM_MODEL` | `mlx-community/Qwen2.5-3B-Instruct-4bit` | Local LLM |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Do not change mid-study |
| `CHUNK_SIZE` | `500` | Words per chunk |
| `CHUNK_OVERLAP` | `80` | Overlap between chunks |
| `MAX_NEW_TOKENS` | `200` | Max generated tokens |
| `TEMPERATURE` | `0` | Greedy decoding for deterministic runs |
| `RANDOM_SEED` | `42` | Set at experiment start (`random` / `numpy` / `mlx`) |
| `EQUIVALENT_INPUT_PRICE_PER_1M` | `0.10` | Equivalent API cost for comparison |
| `EQUIVALENT_OUTPUT_PRICE_PER_1M` | `0.30` | Equivalent API cost for comparison |

Changing `CHUNK_SIZE`, `CHUNK_OVERLAP`, or the PDF requires re-running `build_index.py`.

修改 `CHUNK_SIZE`、`CHUNK_OVERLAP` 或更换 PDF 后，必须重新执行 `build_index.py`。

---

## 7. First-Time Workflow / 首次运行流程

**English:** You must build the vector index **before** running experiments.

**中文：** 运行实验前，**必须先**构建向量索引。

```bash
# Step 1 — Build index (once per book / chunk config)
python scripts/build_index.py
# Expected: Extracted 143 pages; created ~175 chunks.

# Step 2 — Smoke test (3 questions × 5 methods = 15 rows)
python scripts/run_experiments.py --limit 3

# Step 3 — Full baseline run (20 questions × 5 methods = 100 rows)
python scripts/run_experiments.py

# Step 4 — Optional 6th method (120 rows total)
python scripts/run_experiments.py --include-optional
```

Or use the helper script:

```bash
bash run_first_time.sh
```

---

## 8. Run Commands / 运行命令

### Core 5 methods / 默认 5 种方法

```bash
python scripts/run_experiments.py
```

### All 6 methods / 全部 6 种方法

```bash
python scripts/run_experiments.py --include-optional
```

### Single method only / 只跑某一种方法

```bash
python scripts/run_experiments.py --method "Query-Aware + Top-2"
```

Method names must match exactly (case and spacing). Use quotes.

方法名必须完全一致（含大小写与空格），建议加引号。

### Limit questions / 限制题目数量（试跑）

```bash
python scripts/run_experiments.py --limit 3
```

---

## 9. Outputs / 输出结果

Each run creates a **timestamped directory**, e.g. `results/20260726_093512/`:

| File | Description |
|------|-------------|
| `detailed_results.xlsx` | One row per (question × method): tokens, latency, cost, answer, retrieved chunks |
| `summary_results.xlsx` | Aggregated metrics; Token/Latency reduction vs Baseline (Top-8) |
| `run_config.json` | Run timestamp, settings snapshot, dependency versions, methods used |

| 文件 | 说明 |
|------|------|
| `detailed_results.xlsx` | 每题 × 每方法一行：Token、延迟、成本、Answer、Retrieved Chunks |
| `summary_results.xlsx` | 汇总指标；相对 Baseline (Top-8) 的 Token / 延迟降幅 |
| `run_config.json` | 运行时间、配置快照、依赖版本、本次使用的方法 |

**Summary sheets:** Overall Summary, Book Summary, General Summary, Rewrite Summary

**汇总表：** Overall、Book、General、Rewrite 四个 sheet

---

## 10. Manual Scoring / 人工评分

**English**

1. Open `detailed_results.xlsx` in the run directory.
2. Fill column **`Score(0-3)`** for every row:

| Score | Meaning |
|-------|---------|
| 3 | Correct and complete |
| 2 | Mostly correct |
| 1 | Partially correct |
| 0 | Incorrect |

3. Save the file.
4. Regenerate summary:

```bash
python scripts/summarise_results.py
# or
python scripts/summarise_results.py --run-dir 20260726_093512
```

**中文**

1. 打开该次运行目录下的 `detailed_results.xlsx`
2. 在 **`Score(0-3)`** 列逐行打分（3 / 2 / 1 / 0，含义同上）
3. 保存文件
4. 重新生成汇总（命令同上）

---

## 11. Experimental Control / 实验控制原则

**English**

- Warm up the LLM before timed generation.
- Use `temperature=0` and a fixed `RANDOM_SEED` for reproducibility.
- Compare Book, General, and Rewrite questions **separately** in analysis.
- Token and latency reductions in summary are computed against **Baseline (Top-8)**.
- Do not change methodology mid-study; enable or disable strategies via CLI flags only.
- Only one process should access `qdrant_storage/` at a time (close IDE database views if locked).

**中文**

- 正式计时前先 warm-up LLM。
- 固定 `temperature=0` 与 `RANDOM_SEED`，保证可复现。
- 分析时 **分开比较** Book、General、Rewrite 三类题目。
- Summary 中的 Token / 延迟降幅均以 **Baseline (Top-8)** 为基准。
- 研究过程中不改变方法论；仅通过命令行开关不同策略。
- 同一时间只应有一个进程访问 `qdrant_storage/`（若报锁错误，请关闭 IDE 中的数据库连接）。

---

## 12. Rebuilding the Index / 重建索引

Re-run when you change the book, chunk parameters, or embedding model:

```bash
python scripts/build_index.py
```

换书、修改 chunk 参数或 Embedding 模型后执行：

```bash
python scripts/build_index.py
```

---

## 13. Tests / 测试

```bash
PYTHONPATH=. pytest tests/ -q
```
