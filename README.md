# 动态 Context 优化 RAG 实验

本地 RAG 可行性实验：固定模型和问题集，只换 Context 检索策略，看 Token、延迟和回答质量怎么变。结果输出到 Excel，方便打分和写报告。

---

## 1. 实验目的

这是一个课程研究用的本地实验，不是上线项目。

我们想验证：在 LLM 和知识库不变的前提下，能不能通过调整 RAG 的 Context 策略（Top-k 多少、要不要检索、按题型决定是否检索），在**不明显掉质量**的情况下，把 Token 和响应时间压下来。

实验里其他条件都锁死，只改检索策略。会记录 Token、耗时、检索 chunk 数，以及人工打的准确度分。

---

## 2. 实验环境（技术栈）

- **平台**：Apple Silicon，本地跑
- **Python** 3.11
- **LLM**：mlx-lm
- **Embedding**：sentence-transformers
- **向量库**：Qdrant local（`qdrant_storage/`）
- **PDF**：PyMuPDF
- **输出**：pandas + Excel

```bash
cd dynamic-context-experiment
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

首次运行会从 Hugging Face 拉模型，需要网络。

---

## 3. 模型选择

**LLM**：`Qwen2.5-3B-Instruct`（4-bit，`mlx-community/Qwen2.5-3B-Instruct-4bit`）—— Mac 上能跑、够轻。

**Embedding**：`BAAI/bge-small-en-v1.5`—— 全程不换。

**生成参数**（`.env`）：

| 参数 | 值 |
|------|-----|
| `TEMPERATURE` | 0 |
| `MAX_NEW_TOKENS` | 200 |
| `RANDOM_SEED` | 16 |

**切块**：`CHUNK_SIZE=500`，`CHUNK_OVERLAP=80` → 143 页约 175 chunks。

---

## 4. 数据集 / 数据准备

**知识库**：*An Introduction to Statistical Learning with Applications in Python*（ISLP），PDF 前 **143 页**（Ch1–3）。

- 文件：`data/book/ISLR_Python_Pages_1_to_143.pdf`
- 集合名：`islr_python_pages_1_143`

**问题集**：`data/questions/questions.csv`，共 **20 题**

- **Book** × 10：必须能用书内内容回答
- **General** × 5：通用概念，不强制检索
- **Rewrite** × 5：改写句子，看表述不是检索

```bash
python scripts/build_index.py                              # 先建索引
python scripts/run_experiments.py --limit 3                # 试跑
python scripts/run_experiments.py --include-optional       # 正式跑 6 种方法
```

---

## 5. 实验内容

### 六种策略

| 方法 | Top-k | 规则 |
|------|-------|------|
| Baseline RAG | 8 | 每次都检索（baseline） |
| Standard RAG | 4 | 每次都检索 |
| Minimal RAG | 2 | 每次都检索 |
| No RAG | 0 | 不检索 |
| Query-Aware | Book→4 | Book 检索，General/Rewrite 不检索 |
| Query-Aware + Top-2 | Book→2 | 同上，Book 只用 Top-2 |

默认跑前 5 种（100 条）；加 `--include-optional` 跑满 6 种（120 条）。

### 常用命令

```bash
python scripts/run_experiments.py
python scripts/run_experiments.py --include-optional
python scripts/run_experiments.py --method "Query-Aware + Top-2"
python scripts/summarise_results.py --run-dir 20260726_202759
```

每次运行会在 `results/YYYYMMDD_HHMMSS/` 下生成：

- `detailed_results.xlsx` — 明细
- `summary_results.xlsx` — 汇总（打分后 regenerate）
- `run_config.json` — 配置快照

---

## 6. 项目结构

```
dynamic-context-experiment/
├── data/
│   ├── book/                      # PDF knowledge base
│   └── questions/questions.csv    # 20-question benchmark
├── scripts/
│   ├── build_index.py             # PDF → chunks → Qdrant
│   ├── run_experiments.py         # main experiment runner
│   └── summarise_results.py       # rebuild summary after scoring
├── src/
│   ├── config.py
│   ├── experiment.py              # 6 RAG methods & metrics
│   ├── llm_mlx.py
│   ├── vector_store.py
│   ├── reporting.py
│   └── ...
├── qdrant_storage/
├── results/
└── requirements.txt
```

---

## 7. 回答评分标准

在 `detailed_results.xlsx` 的 `Score (0-3)` 列手动打分：

- **3** — 对且完整
- **2** — 大体对，有小问题
- **1** — 只对一部分
- **0** — 错或未答

打完分运行 `python scripts/summarise_results.py` 更新汇总。

---

## 8. 实验结果

> 数据来自 `results/20260726_202759/`（20 题 × 6 方法 = 120 条）

**一句话**：Query-Aware + Top-2 分数最高、Token 也省最多；No RAG 最快但 Book 题掉分明显；chunk 不是越多越好。

### 整体（相对 Baseline Top-8）

| 方法 | Avg Score | Input Tokens | Token↓ | 总耗时↓ |
|------|-----------|-------------|--------|---------|
| Baseline (Top-8) | 2.70 | 4387 | — | — |
| Standard RAG (Top-4) | 2.50 | 2194 | 50% | 45% |
| Minimal RAG (Top-2) | 2.85 | 1155 | 74% | 63% |
| No RAG | 2.55 | 82 | 98% | 88% |
| Query-Aware | 2.75 | 1165 | 73% | 65% |
| **Query-Aware + Top-2** | **2.90** | **654** | **85%** | **73%** |

### 分题型均分

| 方法 | Book | General | Rewrite |
|------|------|---------|---------|
| Baseline | 2.7 | 2.8 | 2.6 |
| Standard | 2.6 | 2.8 | 2.0 |
| Minimal | 2.9 | 3.0 | 2.6 |
| No RAG | 2.2 | 3.0 | 2.8 |
| Query-Aware | 2.6 | 3.0 | 2.8 |
| Query-Aware + Top-2 | 2.9 | 3.0 | 2.8 |

Book 题离不开检索；General / Rewrite 用 Query-Aware 跳过检索，质量和 No RAG 一样但 Book 题仍有检索兜底。

详细数据见 `results/20260726_202759/summary_results.xlsx`。

---

```bash
PYTHONPATH=. pytest tests/ -q
```
