# 动态 Context 优化 RAG 实验

本地 RAG 可行性实验：固定模型和问题集，只换 Context 检索策略，看 Token、延迟和回答质量怎么变。结果输出到 Excel，方便打分和写报告。

---

## 1. 实验目的

比较不同 Context 策略对 RAG 系统的影响，重点观察Input Token 数量、Response Time（Latency）、回答质量（简单人工评分）

---

## 2. 实验环境（技术栈）


| 组件        | 选型                              |
| --------- | ------------------------------- |
| 平台        | MacBook Air（Apple M2，16 GB），本地跑 |
| Python    | 3.11                            |
| LLM       | mlx-lm                          |
| Embedding | sentence-transformers           |
| 向量库       | Qdrant local（`qdrant_storage/`） |
| PDF 解析    | PyMuPDF                         |
| 输出        | pandas + Excel                  |


```bash
pip install -r requirements.txt
```

**主要依赖**（见 `requirements.txt`），首次运行会从 Hugging Face 拉模型，需要网络。

---



## 3. 模型选择


| 类型        | 模型                         | 说明                                                |
| --------- | -------------------------- | ------------------------------------------------- |
| LLM       | Qwen2.5-3B-Instruct（4-bit） | `mlx-community/Qwen2.5-3B-Instruct-4bit`，Mac 本地可跑 |
| Embedding | BAAI/bge-small-en-v1.5     | 全程固定，实验期间不更换                                      |


**生成参数**（`.env`）：


| 参数             | 值   | 说明         |
| -------------- | --- | ---------- |
| TEMPERATURE    | 0   | 贪心解码，保证可复现 |
| MAX_NEW_TOKENS | 200 | 单次最大生成长度   |
| RANDOM_SEED    | 16  | 固定随机种子     |


**切块参数**：


| 参数            | 值          | 说明                   |
| ------------- | ---------- | -------------------- |
| CHUNK_SIZE    | 500        | 每块约 500 词            |
| CHUNK_OVERLAP | 80         | 块间重叠 80 词            |
| 索引规模          | 175 chunks | 143 页 PDF 切块后约 175 块 |


---



## 4. 数据集 / 数据准备

**知识库**


| 项目   | 说明                                                                        |
| ---- | ------------------------------------------------------------------------- |
| 书名   | An Introduction to Statistical Learning with Applications in Python（ISLP） |
| 使用范围 | PDF 前 143 页（Ch1-3）                                                        |
| 文件路径 | `data/book/ISLR_Python_Pages_1_to_143.pdf`                                |
| 集合名  | `islr_python_pages_1_143`                                                 |


**问题集**（`data/questions/questions.csv`，共 20 题）


| 题型      | 数量  | 说明           |
| ------- | --- | ------------ |
| Book    | 10  | 必须能用书内内容回答   |
| General | 5   | 通用概念，不强制检索   |
| Rewrite | 5   | 改写句子，看表述不是检索 |


```bash
python scripts/build_index.py
```

---



## 5. 实验内容



### 5.1 六种 RAG 策略


| 序号  | 方法                  | Top-k  | 规则                          |
| --- | ------------------- | ------ | --------------------------- |
| 1   | Baseline RAG        | 8      | 每次都检索（baseline）             |
| 2   | Standard RAG        | 4      | 每次都检索                       |
| 3   | Minimal RAG         | 2      | 每次都检索                       |
| 4   | No RAG              | 0      | 不检索                         |
| 5   | Query-Aware         | Book 4 | Book 检索，General/Rewrite 不检索 |
| 6   | Query-Aware + Top-2 | Book 2 | 同上，Book 只用 Top-2            |


默认跑前 5 种（100 条）；加 `--include-optional` 跑满 6 种（120 条）。

### 5.2 运行命令

```bash
python scripts/run_experiments.py
python scripts/run_experiments.py --include-optional
python scripts/run_experiments.py --method "Query-Aware + Top-2"
python scripts/summarise_results.py --run-dir 20260726_202759
```



### 5.3 输出文件

每次运行会在 `results/YYYYMMDD_HHMMSS/` 下生成：


| 文件                      | 说明          |
| ----------------------- | ----------- |
| `detailed_results.xlsx` | 每题 x 每方法明细  |
| `summary_results.xlsx`  | 汇总（打分后重新生成） |
| `run_config.json`       | 配置快照        |




### 5.4 实验流程

1. `build_index.py` 构建向量索引
2. `run_experiments.py` 跑实验并生成明细
3. 在 Excel 中手动打分
4. `summarise_results.py` 重新生成汇总

---



## 6. 项目结构

```
dynamic-context-experiment/
├── data/
│   ├── book/                         # ISLR PDF knowledge base
│   └── questions/
│       └── questions.csv             # 20-question benchmark
├── scripts/
│   ├── _bootstrap.py                 # add project root to sys.path
│   ├── build_index.py                # PDF to chunks to Qdrant
│   ├── run_experiments.py             # main experiment runner
│   └── summarise_results.py          # rebuild summary after scoring
├── src/
│   ├── config.py                     # load settings from .env
│   ├── models.py                     # Chunk, ExperimentRow dataclasses
│   ├── pdf_loader.py                 # extract text and chunk PDF pages
│   ├── vector_store.py               # embedding and Qdrant search
│   ├── prompts.py                    # system prompt and prompt builder
│   ├── llm_mlx.py                    # local Qwen inference via mlx-lm
│   ├── experiment.py                 # 6 RAG methods and metric collection
│   ├── reporting.py                  # build summary Excel workbooks
│   └── run_metadata.py               # timestamped run dirs and run_config.json
├── tests/
│   ├── test_logic.py                 # method routing and experiment logic
│   ├── test_reporting.py             # summary and score column handling
│   └── test_run_metadata.py          # run directory helpers
├── qdrant_storage/                   # local vector index (generated)
├── results/                          # timestamped experiment outputs
├── .env.example                      # config template
├── run_first_time.sh                 # first-time setup script
├── requirements.txt                  # Python dependencies
└── README.md
```

---



## 7. 回答评分标准

在 `detailed_results.xlsx` 的 Score (0-3) 列手动打分：


| 分数  | 含义       |
| --- | -------- |
| 3   | 对且完整     |
| 2   | 大体对，有小问题 |
| 1   | 只对一部分    |
| 0   | 错或未答     |


打完分运行：

```bash
python scripts/summarise_results.py
```

---



## 8. 实验结果

数据来自 `results/20260726_202759/`（20 题 x 6 方法 = 120 条）

结论：**Query-Aware + Top-2** 分数最高、Token 也省最多；No RAG 最快但 Book 题掉分明显；chunk 不是越多越好。

### 整体（相对 Baseline Top-8）


| 方法                   | Avg Score | Input Tokens | Token 降幅 | 总耗时降幅 |
| -------------------- | --------- | ------------ | -------- | ----- |
| Baseline (Top-8)     | 2.70      | 4387         | -        | -     |
| Standard RAG (Top-4) | 2.50      | 2194         | 50%      | 45%   |
| Minimal RAG (Top-2)  | 2.85      | 1155         | 74%      | 63%   |
| No RAG               | 2.55      | 82           | 98%      | 88%   |
| Query-Aware          | 2.75      | 1165         | 73%      | 65%   |
| **Query-Aware + Top-2** | **2.90** | **654** | **85%** | **73%** |




### 分题型均分


| 方法                  | Book | General | Rewrite |
| ------------------- | ---- | ------- | ------- |
| Baseline            | 2.7  | 2.8     | 2.6     |
| Standard            | 2.6  | 2.8     | 2.0     |
| Minimal             | **2.9** | 3.0  | 2.6     |
| No RAG              | 2.2  | 3.0     | 2.8     |
| Query-Aware         | 2.6  | 3.0     | 2.8     |
| **Query-Aware + Top-2** | **2.9** | **3.0** | **2.8** |


Book 题离不开检索；General / Rewrite 用 Query-Aware 跳过检索，质量和 No RAG 一样但 Book 题仍有检索兜底。

详细数据见 `results/20260726_202759/summary_results.xlsx`。