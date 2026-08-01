# 动态 Context 优化 RAG 实验

---

## 介绍

比较不同 Context 检索策略对 RAG 系统的影响，重点观察输入 Token 数量、响应延迟（Latency）与回答质量（0–3 人工评分）。

**技术栈**

| 组件 | 选型 |
| --- | --- |
| 运行平台 | MacBook Air M2，16 GB，本地运行 |
| Python | 3.11 |
| LLM 推理 | `mlx-lm` |
| Embedding | `sentence-transformers` |
| 向量库 | Qdrant Local（`qdrant_storage/`） |
| PDF 解析 | PyMuPDF |
| 结果处理 | pandas + openpyxl |

```bash
pip install -r requirements.txt
```

首次运行会从 Hugging Face 拉取 LLM 与 Embedding 模型，需要网络。

---

## 配置

复制 `.env.example` 为 `.env` 后可按需修改。主要项如下。

### 模型

| 类型 | 模型 | 说明 |
| --- | --- | --- |
| LLM | Qwen2.5-3B-Instruct-4bit | `mlx-community/Qwen2.5-3B-Instruct-4bit` |
| Embedding | BAAI/bge-small-zh-v1.5 | 所有方法固定使用同一模型 |

### 生成参数

| 参数 | 值 | 说明 |
| --- | --- | --- |
| temperature | 0 | 减少随机性 |
| max_new_tokens | 200 | 最大输出长度 |
| seed | 16 | 固定随机种子 |

### 索引切块

参数见 `pdf_loader.py` / `.env`。

| 参数 | 值 | 说明 |
| --- | --- | --- |
| CHUNK_SIZE | 500 | 按字符切分，每块约 500 字 |
| CHUNK_OVERLAP | 80 | 块间重叠约 80 字 |
| collection_name | huozhe | Qdrant 集合名 |

> 实验中的 **Top-k** 指检索返回的 chunk **数量**，与 CHUNK_SIZE 是不同概念。

### 知识库与问题集

| 项目 | 路径 |
| --- | --- |
| 知识库 | `data/book/活着.pdf` |
| 问题集 | `data/questions/questions.csv`（20 题：Book 10 / General 5 / Rewrite 5） |

---

## 项目结构

```
dynamic-context-experiment/
├── data/
│   ├── book/                         # 《活着》PDF 知识库
│   └── questions/
│       └── questions.csv             # 20 题基准问题集
├── scripts/
│   ├── _bootstrap.py                 # 脚本运行时加入项目根目录到 sys.path
│   ├── build_index.py                # PDF → chunks → Qdrant
│   ├── run_experiments.py            # 主实验入口
│   └── summarise_results.py          # 打分后重新生成汇总
├── src/
│   ├── config.py                     # 从 .env 加载配置
│   ├── models.py                     # Chunk、ExperimentRow 等数据结构
│   ├── pdf_loader.py                 # PDF 提取与字符切块
│   ├── vector_store.py               # Embedding 与 Qdrant 检索
│   ├── prompts.py                    # 中文 system / user prompt
│   ├── llm_mlx.py                    # 本地 Qwen 推理（mlx-lm）
│   ├── experiment.py                 # 6 种 RAG 方法与指标采集
│   ├── reporting.py                  # 汇总 Excel
│   └── run_metadata.py               # 时间戳 run 目录与 run_config.json
├── tests/
│   ├── test_logic.py                 # 方法路由与实验逻辑
│   ├── test_pdf_loader.py            # 字符切块
│   ├── test_reporting.py             # 汇总与 Score 列
│   └── test_run_metadata.py          # run 目录辅助函数
├── qdrant_storage/                   # 本地向量索引（build_index 生成）
├── results/                          # 实验输出（按时间戳分子目录）
├── .env.example                      # 配置模板
├── run_first_time.sh                 # 首次环境搭建脚本
├── requirements.txt
└── README.md
```
