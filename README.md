# 动态 Context 优化 RAG 实验

本项目是一个可复现的本地 RAG 可行性实验框架，用于比较不同 Context 检索策略对 Token 用量、响应延迟、估算成本与回答质量的影响。实验在 Apple Silicon 上本地运行，固定 LLM、Embedding、切块参数与生成参数，每次只改变检索策略，并将结果输出为 Excel 与 JSON，便于论文分析与人工评分。

---

## 1. 实验目的

本实验属于学术研究用的可行性验证，**不是生产系统**。核心问题是：

> 在固定模型与知识库的前提下，动态调整 Context 检索策略，能否在**不显著牺牲回答质量**的情况下，降低 Token 消耗与响应延迟？

实验设计遵循控制变量原则：

- **固定项**：LLM、Embedding 模型、PDF 切块参数、20 题问题集、生成参数（`temperature=0`，`max_new_tokens=200`）
- **变量项**：六种 RAG Context 检索策略（Top-k 数量、是否检索、是否按题型路由）

采集指标包括：

| 指标 | 说明 |
|------|------|
| Input / Output / Total Tokens | 输入、输出与总 Token 数 |
| Retrieval / LLM / Total Time | 检索、生成与总耗时（毫秒） |
| Output Tokens/sec | 生成吞吐 |
| Estimated Cost (USD) | 等效 API 成本（本地推理免费，用于跨方法对比） |
| Retrieved Chunks | 实际检索到的 chunk 数量 |
| Score (0–3) | 人工准确度评分 |

---

## 2. 实验环境（技术栈）

| 组件 | 选型 |
|------|------|
| 硬件平台 | Apple Silicon（Mac，本地推理） |
| 编程语言 | Python 3.11 |
| LLM 推理 | mlx-lm（MLX 框架） |
| Embedding | sentence-transformers |
| 向量数据库 | Qdrant 本地模式（`qdrant_storage/`） |
| PDF 解析 | PyMuPDF |
| 数据处理与输出 | pandas、openpyxl |
| 配置管理 | python-dotenv（`.env`） |

### 环境安装

```bash
cd dynamic-context-experiment
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

首次运行会从 Hugging Face 下载 LLM 与 Embedding 模型，请确保网络可用。

---

## 3. 模型选择

### 3.1 大语言模型（LLM）

| 项目 | 配置 |
|------|------|
| 模型 | **Qwen2.5-3B-Instruct**（4-bit 量化） |
| Hugging Face ID | `mlx-community/Qwen2.5-3B-Instruct-4bit` |
| 选择理由 | 可在 Apple Silicon 上本地运行，体量适中，适合学术实验的可复现性与成本控制 |

### 3.2 Embedding 模型

| 项目 | 配置 |
|------|------|
| 模型 | **BAAI/bge-small-en-v1.5** |
| 选择理由 | 英文语义检索效果稳定、体积较小；**全程固定，实验期间不更换** |

### 3.3 生成参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `TEMPERATURE` | `0` | 贪心解码，保证可复现 |
| `MAX_NEW_TOKENS` | `200` | 单次最大生成长度 |
| `RANDOM_SEED` | `16`（可配置） | 固定 random / numpy / mlx 随机种子 |

### 3.4 切块与成本参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `CHUNK_SIZE` | `500` | 每块约 500 词 |
| `CHUNK_OVERLAP` | `80` | 块间重叠 80 词 |
| `EQUIVALENT_INPUT_PRICE_PER_1M` | `0.10` | 等效输入 Token 单价（USD/百万） |
| `EQUIVALENT_OUTPUT_PRICE_PER_1M` | `0.30` | 等效输出 Token 单价（USD/百万） |

---

## 4. 数据集 / 数据准备

### 4.1 知识库

| 项目 | 说明 |
|------|------|
| 书名 | *An Introduction to Statistical Learning with Applications in Python*（ISLP） |
| 使用范围 | PDF **前 143 页**（第 1–3 章：Introduction、Statistical Learning、Linear Regression） |
| 文件路径 | `data/book/ISLR_Python_Pages_1_to_143.pdf` |
| 向量集合名 | `islr_python_pages_1_143` |
| 索引规模 | 143 页 → 约 **175 个 chunk** |

### 4.2 问题集

| 项目 | 说明 |
|------|------|
| 文件路径 | `data/questions/questions.csv` |
| 题目总数 | **20 题** |
| 题型分布 | 10 Book + 5 General + 5 Rewrite |

| 题型 | 数量 | 说明 |
|------|------|------|
| **Book** | 10 | 必须能依据书内内容作答 |
| **General** | 5 | 通用概念题，不强制依赖书本 |
| **Rewrite** | 5 | 改写/润色题，考察表述而非检索 |

### 4.3 数据准备步骤

```bash
# 1. 构建向量索引（换书或改 chunk 参数后必须重跑）
python scripts/build_index.py

# 2. 试跑（3 题 × 方法数）
python scripts/run_experiments.py --limit 3

# 3. 正式实验（20 题 × 6 方法 = 120 条）
python scripts/run_experiments.py --include-optional
```

或使用一键脚本：

```bash
bash run_first_time.sh
```

---

## 5. 实验内容

### 5.1 六种 RAG 策略

| # | 方法 | Top-k | 检索规则 |
|---|------|-------|----------|
| 1 | Baseline RAG | 8 | 始终检索（对照组） |
| 2 | Standard RAG | 4 | 始终检索 |
| 3 | Minimal RAG | 2 | 始终检索 |
| 4 | No RAG | 0 | 始终不检索 |
| 5 | Query-Aware | Book 用 4 | Book 检索；General / Rewrite 不检索 |
| 6 | Query-Aware + Top-2 | Book 用 2 | 同 Query-Aware，Book 题仅用 Top-2 |

- 默认运行前 5 种方法（100 条 = 20 题 × 5 方法）
- 加 `--include-optional` 运行全部 6 种（120 条）

### 5.2 常用运行命令

```bash
# 默认 5 种方法
python scripts/run_experiments.py

# 全部 6 种方法
python scripts/run_experiments.py --include-optional

# 只跑某一种方法
python scripts/run_experiments.py --method "Query-Aware + Top-2"

# 限制题目数量（试跑）
python scripts/run_experiments.py --limit 3
```

方法名必须完全一致（含大小写与空格），建议加引号。

### 5.3 实验控制原则

- 正式计时前先 warm-up LLM
- 固定 `temperature=0` 与 `RANDOM_SEED`
- 分析时**分开比较** Book、General、Rewrite 三类题目
- Summary 中 Token / 延迟降幅均以 **Baseline (Top-8)** 为基准
- 同一时间只应有一个进程访问 `qdrant_storage/`（若报锁错误，请关闭 IDE 中的 Qdrant 数据库连接）

### 5.4 输出文件

每次运行生成时间戳目录，例如 `results/20260726_202759/`：

| 文件 | 说明 |
|------|------|
| `detailed_results.xlsx` | 每题 × 每方法一行：Token、延迟、成本、Answer、Retrieved Chunks |
| `summary_results.xlsx` | 汇总指标（Overall / Book / General / Rewrite 四个 sheet） |
| `run_config.json` | 运行时间、配置快照、依赖版本、本次使用的方法 |

人工打分后重新生成汇总：

```bash
python scripts/summarise_results.py
python scripts/summarise_results.py --run-dir 20260726_202759
```

---

## 6. 项目结构

```
dynamic-context-experiment/
├── data/
│   ├── book/                         # 知识库 PDF
│   └── questions/questions.csv       # 固定 20 题基准
├── scripts/
│   ├── build_index.py                # PDF → chunks → Qdrant
│   ├── run_experiments.py            # 实验主入口
│   ├── summarise_results.py          # 人工打分后重新生成汇总
│   └── _bootstrap.py                 # 修复 scripts 模块导入
├── src/
│   ├── config.py                     # 读取 .env 配置
│   ├── pdf_loader.py                 # PDF 提取与切块
│   ├── vector_store.py               # Embedding + Qdrant 检索
│   ├── prompts.py                    # Prompt 模板
│   ├── llm_mlx.py                    # 本地 Qwen 推理
│   ├── experiment.py                 # 实验主循环与六种方法
│   ├── reporting.py                  # 汇总 Excel 生成
│   ├── run_metadata.py               # 时间戳目录与 run_config.json
│   └── models.py                     # 数据结构定义
├── qdrant_storage/                   # 本地向量索引（自动生成）
├── results/                          # 按时间戳存放实验结果
├── tests/                            # 单元测试
├── .env.example                      # 配置模板
├── run_first_time.sh                 # 首次运行脚本
└── requirements.txt
```

---

## 7. 回答评分标准

评分在 `detailed_results.xlsx` 的 **`Score (0-3)`** 列（或 `Score(0-3)`）中手动填写，对照 `Ground Truth` 与题型要求逐行评判。

| 分数 | 含义 |
|------|------|
| **3** | 正确且完整 |
| **2** | 大体正确，有小缺漏或表述不够精确 |
| **1** | 部分正确，关键信息缺失或有明显错误 |
| **0** | 错误、未作答、或未完成题目要求（如 Rewrite 题未改写） |

### 分题型评分要点

| 题型 | 评分侧重 |
|------|----------|
| **Book** | 是否准确反映书内概念；拒答或仅引用页码而无实质解释应低分 |
| **General** | 是否直接回答概念，而非偏离到教材介绍 |
| **Rewrite** | 是否按要求简化、润色或改写，而非照抄原句 |

### 评分工作流

1. 打开运行目录下的 `detailed_results.xlsx`
2. 在 `Score (0-3)` 列逐行打分，可在 `Notes` 列记录理由
3. 保存文件
4. 运行 `python scripts/summarise_results.py` 更新 `summary_results.xlsx`

---

## 8. 实验结果

> 数据来源：`results/20260726_202759/`（2026-07-26，20 题 × 6 方法 = **120 条**，含人工评分）

### 8.1 总体结论

1. **Token 与延迟**：相对 Baseline (Top-8)，No RAG 降幅最大，Query-Aware + Top-2 在保留检索能力的方法中效率最高。
2. **回答质量**：Query-Aware + Top-2 均分最高（2.90），Minimal RAG 次之（2.85）；**chunk 数量并非越多越好**。
3. **Query-Aware 有效性**：General / Rewrite 题不检索仍保持 3.0 分，Book 题与 Standard RAG 相当，说明按题型路由可行。
4. **Book 题依赖检索**：No RAG 在 Book 题均分仅 2.2，部分题目（如 Q01、Q10）得 0 分，证明书本 Context 对 Book 题不可或缺。

### 8.2 整体汇总（20 题）

| 方法 | 均分 | 平均 Input Tokens | Token 降幅 | 平均总耗时 (ms) | 延迟降幅 | 估算成本 (USD) |
|------|------|-------------------|------------|-----------------|----------|----------------|
| Baseline (Top-8) | 2.70 | 4387 | — | 33247 | — | 0.000453 |
| Standard RAG (Top-4) | 2.50 | 2194 | 50.0% | 18452 | 44.5% | 0.000234 |
| Minimal RAG (Top-2) | 2.85 | 1155 | 73.7% | 12234 | 63.2% | 0.000131 |
| No RAG | 2.55 | 82 | 98.1% | 3880 | 88.3% | 0.000021 |
| Query-Aware | 2.75 | 1165 | 73.4% | 11691 | 64.8% | 0.000130 |
| **Query-Aware + Top-2** | **2.90** | **654** | **85.1%** | **8860** | **73.3%** | **0.000080** |

*Token 降幅与延迟降幅均以 Baseline (Top-8) 为基准。*

### 8.3 分题型均分

| 方法 | Book (10题) | General (5题) | Rewrite (5题) |
|------|-------------|---------------|---------------|
| Baseline (Top-8) | 2.7 | 2.8 | 2.6 |
| Standard RAG (Top-4) | 2.6 | 2.8 | 2.0 |
| Minimal RAG (Top-2) | 2.9 | 3.0 | 2.6 |
| No RAG | 2.2 | 3.0 | 2.8 |
| Query-Aware | 2.6 | 3.0 | 2.8 |
| Query-Aware + Top-2 | 2.9 | 3.0 | 2.8 |

**解读：**

- **Book**：Minimal 与 Query-Aware + Top-2 均分最高（2.9），No RAG 最低（2.2）
- **General**：Minimal、No RAG、Query-Aware 系列均为 3.0；有 RAG 的 Baseline / Standard 因回答偏教材内容而为 2.8
- **Rewrite**：Standard RAG 仅 2.0（Q16 照抄原句得 0 分）；其余方法 2.6–2.8

### 8.4 关键发现

| 发现 | 说明 |
|------|------|
| 效率与质量可兼顾 | Query-Aware + Top-2 在 85% Token 降幅下均分最高，优于 Baseline |
| Top-8 非最优 | Baseline 均分 2.70，低于 Minimal (2.85) 与 Query-Aware + Top-2 (2.90) |
| 无检索的代价 | No RAG 效率最高但 Book 题质量明显下降，不适合作为统一策略 |
| Query-Aware 路由合理 | General / Rewrite 跳过检索后 Token 接近 No RAG，质量保持 3.0 |
| 失败模式可解释 | Standard RAG 低分主要来自 Q06（仅引用页码）与 Q16（Rewrite 未改写） |

### 8.5 结果文件

| 文件 | 路径 |
|------|------|
| 详细结果 | `results/20260726_202759/detailed_results.xlsx` |
| 汇总结果 | `results/20260726_202759/summary_results.xlsx` |
| 运行配置 | `results/20260726_202759/run_config.json` |

---

## 测试

```bash
PYTHONPATH=. pytest tests/ -q
```
