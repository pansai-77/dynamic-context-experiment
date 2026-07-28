# 动态 Context 优化 RAG 实验

---

## 1. 实验目的

比较不同 Context 策略对 RAG 系统的影响，重点观察输入 Token 数量、响应延迟（Latency）与回答质量（0–3 人工评分）。

---

## 2. 实验环境与技术栈


| 组件           | 实际选型                            |
| ------------ | ------------------------------- |
| 运行平台         | MacBook Air M2，16 GB，本地运行       |
| Python       | 3.11                            |
| LLM 推理框架     | `mlx-lm`                        |
| Embedding 框架 | `sentence-transformers`         |
| 向量库          | Qdrant Local（`qdrant_storage/`） |
| PDF 解析       | PyMuPDF                         |
| 结果处理         | pandas + openpyxl               |


**主要依赖**（完整列表见 `requirements.txt`）：

```text
transformers
sentence-transformers
torch
qdrant-client
mlx-lm
PyMuPDF
pandas
openpyxl
```

```bash
pip install -r requirements.txt
```

首次运行会从 Hugging Face 拉取 LLM 与 Embedding 模型，需要网络。`torch` 由 `sentence-transformers` 安装时自动引入。

---



## 3. 模型与参数配置


| 类型        | 模型                       | 说明                                                          |
| --------- | ------------------------ | ----------------------------------------------------------- |
| LLM       | Qwen2.5-3B-Instruct-4bit | 通过 MLX 在 Mac 本地运行（`mlx-community/Qwen2.5-3B-Instruct-4bit`） |
| Embedding | BAAI/bge-small-en-v1.5   | 所有方法固定使用同一模型                                                |



| 参数             | 值   | 说明           |
| -------------- | --- | ------------ |
| temperature    | 0   | 减少随机性，使输出更稳定 |
| max_new_tokens | 200 | 限制最大输出长度     |
| seed           | 16  | 固定随机种子       |


**索引切块参数**（PDF → 向量库，见 `pdf_loader.py`）：


| 参数            | 值          | 说明                                         |
| ------------- | ---------- | ------------------------------------------ |
| CHUNK_SIZE    | 500        | 每页文本按空白分词切分，每块约 500 词（非 tokenizer Token 数） |
| CHUNK_OVERLAP | 80         | 块间重叠约 80 词；每页独立滑动窗口切块                      |
| 索引规模          | 175 chunks | 143 页 PDF 切块后约 175 块                       |


> 本实验中的 **Top-k** 指检索时返回的 chunk **数量**，与上述索引切块大小（CHUNK_SIZE）是不同概念。

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


| 题型      | 数量  | 说明                             |
| ------- | --- | ------------------------------ |
| Book    | 10  | 答案可直接从所提供的 PDF 范围内获得，用于测试检索能力  |
| General | 5   | 通用知识问题，用于测试是否需要检索              |
| Rewrite | 5   | 文本改写任务，用于测试无关 Context 是否影响指令执行 |


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

数据来自 `results/20260726_202759/`（20 题 x 6 方法 = 120 条，相对 Baseline Top-8 计算降幅）。

### 最终统计表


| Method                  | Avg Input Tokens | Token Reduction | Avg Total Time (ms) | Latency Reduction | Avg Score | 备注                              |
| ----------------------- | ---------------- | --------------- | ------------------- | ----------------- | --------- | ------------------------------- |
| Baseline (Top-8)        | 4387             | -               | 33247               | -                 | 2.70      | 对照组                             |
| Standard RAG (Top-4)    | 2194             | 50%             | 18452               | 45%               | 2.50      | Token 减半，但均分最低                  |
| Minimal RAG (Top-2)     | 1155             | 74%             | 12234               | 63%               | 2.85      | 检索 Top-2，质量较好                   |
| No RAG                  | 82               | 98%             | 3880                | 88%               | 2.55      | 输入 Token 最少，Book 题掉分明显          |
| Query-Aware             | 1165             | 73%             | 11691               | 65%               | 2.75      | 按题型决定是否检索                       |
| **Query-Aware + Top-2** | **654**          | **85%**         | **8860**            | **73%**           | **2.90**  | **综合最优：平均分最高，同时大幅降低 Token 与延迟** |




### 结论

**Query-Aware + Top-2** 在本实验中最佳。与 Baseline Top-8 相比，它将平均输入 Token 减少 85%，平均总延迟减少 73%，同时获得最高平均质量分 2.90。**No RAG** 虽然速度最快、Token 使用最少（减少 98%），但其 Book QA 平均分仅为 2.20，说明书本问题仍然需要检索。对于 General 和 Rewrite 任务，无检索方法取得了相同或更高的质量分，表明这两类任务不需要额外 Context。在本实验中，增加检索 Chunk 数量没有进一步提高回答质量；Top-2 已能提供足够的相关 Context，而额外 chunk 可能引入无关信息（例如 Standard Top-4 均分 2.50，为各方法最低）。

### 其他-分题型均分


| Method                  | Book    | General | Rewrite |
| ----------------------- | ------- | ------- | ------- |
| Baseline                | 2.7     | 2.8     | 2.6     |
| Standard                | 2.6     | 2.8     | 2.0     |
| Minimal                 | 2.9     | 3.0     | 2.6     |
| No RAG                  | 2.2     | 3.0     | 2.8     |
| Query-Aware             | 2.6     | 3.0     | 2.8     |
| **Query-Aware + Top-2** | **2.9** | **3.0** | **2.8** |


详细数据见 `results/20260726_202759/summary_results.xlsx`。