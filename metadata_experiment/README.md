# 实验二：LLM 生成 Metadata RAG（Phase 1）

---

## 介绍

在实验一选定的 **Query-Aware Top-4** 基线上，比较 **是否使用 Metadata Filter** 对 RAG 检索效率与回答质量的影响。

索引阶段由本地 LLM（Qwen2.5-3B）为每个 chunk 离线生成 metadata（人物、topic、关键词）；检索阶段对 Book 题先做 **Top-2 Topic 路由**，再对 chunk 的 `topics` 字段做 **OR 过滤**，最后在过滤后的候选集内取向量 Top-4。

**与实验一的关系**

| 项目 | 实验一 | 实验二 |
| --- | --- | --- |
| 目录 | 根目录 `src/`、`scripts/` | `metadata_experiment/` |
| Index metadata | `qdrant_storage/index_metadata.json` | `metadata_experiment/index_metadata.json` |
| Qdrant collection | `huozhe` | `huozhe_meta` |
| 结果目录 | `results/` | `metadata_experiment/results/` |

共用：`data/book/`、`data/questions/questions.csv`、`requirements.txt`、`.env`、根目录 `qdrant_storage/`。

环境安装与根目录 README 相同：

```bash
cp .env.example .env
pip install -r requirements.txt
```

---

## 研究假设

- **H1：** Metadata Filter 预期降低检索延迟（Latency）。
- **H2：** Metadata Filter 预期减少输入 Token 数量。
- **H3：** Metadata Filter 在回答质量上与 Baseline 相当。

---

## 实验方法

| 方法 | 说明 |
| --- | --- |
| **A：Query-Aware Top-4** | Book 题 → 全库向量 Top-4（无 Metadata Filter） |
| **B：Query-Aware + Metadata Top-4** | Book 题 → Top-2 Topic 路由 → OR Metadata Filter → Top-4 |

General / Rewrite 题在两种方法下均 **跳过检索**，直接调用 LLM。

**正式主实验（frozen）：** Top-2 Router → Top-1 ∪ Top-2（OR Filter）→ Top-4，**不使用** `--with-fallback`。

**补充实验（非主实验）：** `--with-fallback` 在 filter 返回 0 条时，将 topic 从 Top-2 扩至 Top-3/4。

---

## 主要指标

与指导书对齐的 Primary 指标：

| 指标 | Detailed Results | Summary |
| --- | --- | --- |
| Retrieval / LLM / Total Time | 每行 | 取平均 |
| Input / Output / Total Tokens | 每行 | Book 题平均 |
| Retrieved Chunks | 每行（General/Rewrite 为 0） | — |
| 回答质量 | `Score(0-3)` 人工打分 | Book Score、Overall Score |

Phase 1 **不使用** gold chunk、Hit@4、MRR 作为主指标。

> **Latency 说明：** 当前实现为每题 × 每方法 **单次检索** 计时，Summary 对 Book 题取描述性平均。可报告数值，但不宜写「显著更快」或作严格性能结论。

---

## 配置

实验二在根目录 `.env` 基础上增加以下项（见 `.env.example`）：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `METADATA_COLLECTION` | `huozhe_meta` | Qdrant 集合名 |
| `METADATA_INDEX_METADATA_PATH` | `metadata_experiment/index_metadata.json` | 索引元数据路径 |
| `METADATA_RESULTS_DIR` | `metadata_experiment/results` | 实验结果目录 |
| `TOPIC_ROUTING_TOP_N` | `2` | Router 取 Top-N topic |
| `METADATA_GEN_MAX_RETRIES` | `2` | 建索引时 metadata 生成失败重试次数 |
| `METADATA_MAX_NEW_TOKENS` | `384` | metadata 生成最大输出 token（与 QA 的 `MAX_NEW_TOKENS=200` 分离） |

其余 LLM、Embedding、切块参数与实验一相同（`continuous_sentence_aware`，target 600 / max 800 / overlap 100）。

| 项目 | 路径 |
| --- | --- |
| Topic 定义 | `metadata_experiment/data/allowed_topics.json`（v3.1） |
| Acceptance 样本 | `metadata_experiment/data/metadata_acceptance_samples.json` |
| Topic 向量缓存 | `metadata_experiment/data/topic_embeddings.json` |

---

## Metadata v3（当前）

- **Ontology：** 7 个事件级 topic — `war`、`politics`、`gambling`、`family`、`medical`、`labor`、`livelihood`
- **Prompt：** 短规则 + few-shot；`topics` 取 0 或 1 个；`keywords` 固定 3 个；无 `importance`
- **Pilot v2.1 索引：** 仅作诊断，**不用于** 正式 H1–H3

### Metadata 语义

| 字段 / 状态 | 含义 |
| --- | --- |
| `topics=[]` 且 `metadata_status="ok"` | 片段无足够清晰的事件 topic（旁白、过渡等） |
| `metadata_status="failed"` | JSON 解析或字段校验失败 |
| Acceptance | 使用 `acceptable_topics`（允许多组合法标签），非单一 gold label |

---

## 推荐执行顺序

1. 保留 Pilot v2.1 索引与报告（仅诊断）。
2. 在 40-chunk 开发集上跑 **v3 acceptance**；目标 pass rate **≥70%** 后再全量重建。
3. 全量建索引 → 查看 topic coverage → filter stats →（建议）Top-4 spot-check。
4. 跑正式 A/B（不加 `--with-fallback`）。
5. 在 `detailed_results.xlsx` 填写 `Score(0-3)` → 重新生成 Summary。

```text
Pilot v2.1（保留）
→ run_metadata_acceptance.py
→ analyze_acceptance.py
→ build_metadata_index.py
→ topic_coverage_report.json
→ analyze_filter_stats.py
→ Top-4 spot-check（建议）
→ run_metadata_experiment.py
→ summarise_results.py
```

---

## 命令

在**仓库根目录**执行。

### 1. Metadata 验收（不写 Qdrant）

```bash
python metadata_experiment/scripts/run_metadata_acceptance.py
python metadata_experiment/scripts/analyze_acceptance.py
```

### 2. 全量建索引（LLM metadata + 向量）

```bash
python metadata_experiment/scripts/build_metadata_index.py
```

查看 topic 分布：

```bash
cat metadata_experiment/topic_coverage_report.json
```

### 3. Filter 统计（10 道 Book 题）

```bash
python metadata_experiment/scripts/analyze_filter_stats.py
python metadata_experiment/scripts/analyze_filter_stats.py \
  --output metadata_experiment/filter_stats.json
```

### 4. 正式 A/B 实验

```bash
python metadata_experiment/scripts/run_metadata_experiment.py
```

只跑某一方法（可选）：

```bash
python metadata_experiment/scripts/run_metadata_experiment.py \
  --method "Query-Aware Top-4"

python metadata_experiment/scripts/run_metadata_experiment.py \
  --method "Query-Aware + Metadata Top-4"
```

只跑部分题目（调试，可选）：

```bash
python metadata_experiment/scripts/run_metadata_experiment.py \
  --question-ids Q01,Q02,Q03
```

### 5. 打分后重算 Summary

```bash
python metadata_experiment/scripts/summarise_results.py --run-dir 20260802_120000
```

`--run-dir` 填 `metadata_experiment/results/` 下的时间戳目录名。

### 6. 补充实验（非主实验）

```bash
python metadata_experiment/scripts/run_metadata_experiment.py --with-fallback
```

### 7. 测试

```bash
pytest metadata_experiment/tests/ -q
```

---

## 输出文件

**建索引：**

| 文件 | 说明 |
| --- | --- |
| `metadata_experiment/index_build_report.json` | 建索引统计 |
| `metadata_experiment/topic_coverage_report.json` | 各 topic 覆盖数量与占比 |
| `metadata_experiment/index_metadata.json` | 索引元数据（切块参数、模型等） |
| `metadata_experiment/data/topic_embeddings.json` | Topic 定义 embedding 缓存 |

**实验结果**（`metadata_experiment/results/{timestamp}/`）：

| 文件 | 说明 |
| --- | --- |
| `detailed_results.xlsx` | 每题 × 每方法明细（Answer、Token、Time、Score） |
| `summary_results.xlsx` | 按方法汇总 |
| `run_config.json` | 本次 run 配置快照 |

**Acceptance**（`metadata_experiment/results/acceptance_*.json`）：40-chunk 标注验收报告。

> **说明：** 当前 `detailed_results.xlsx` 不保存 Predicted Top-1/Top-2 Topic 与 retrieved chunk ID；排查检索问题时需借助 `analyze_filter_stats.py` 或单独 spot-check。

---

## 项目结构

```
metadata_experiment/
├── data/
│   ├── allowed_topics.json           # v3.1 Topic Ontology
│   ├── metadata_acceptance_samples.json  # 40-chunk 验收集
│   └── topic_embeddings.json         # Router 用 topic 向量缓存
├── scripts/
│   ├── _bootstrap.py
│   ├── build_metadata_index.py       # 全量建索引
│   ├── run_metadata_acceptance.py    # Acceptance 生成（不写 Qdrant）
│   ├── analyze_acceptance.py         # Pass rate + 混淆矩阵
│   ├── analyze_filter_stats.py       # Book 题 filter 缩圈统计
│   ├── run_metadata_experiment.py      # 正式 A/B 入口
│   └── summarise_results.py          # 打分后重算 Summary
├── src/
│   ├── config.py                     # 实验二配置（读根目录 .env）
│   ├── models.py                     # ChunkMetadata、ExperimentRow 等
│   ├── prompts.py                    # Metadata 生成 prompt + few-shot
│   ├── metadata_generator.py         # 单 chunk metadata 生成与重试
│   ├── metadata_parsing.py             # JSON 解析与字段规范化
│   ├── metadata_retriever.py         # Qdrant 检索 + Metadata Filter
│   ├── topic_router.py               # Question → Top-N Topic（embedding）
│   ├── topic_coverage.py             # 索引 topic 分布报告
│   ├── filter_stats.py               # Filter 统计分析
│   ├── acceptance_analysis.py        # Acceptance 分析逻辑
│   ├── experiment.py                 # 主实验循环与指标采集
│   ├── logic.py                      # 方法定义（A/B）
│   ├── reporting.py                  # Excel 列规范与 Summary
│   └── index_metadata.py             # 索引元数据校验
├── tests/
├── results/                          # 实验与 acceptance 输出
├── index_metadata.json
├── index_build_report.json
├── topic_coverage_report.json
└── README.md
```

---

## 与实验一的隔离

- 实验二使用独立 collection **`huozhe_meta`**，不覆盖实验一的 **`huozhe`**。
- 实验二代码与结果均在 `metadata_experiment/` 下，不修改根目录实验一源码。
- 正式 H1–H3 仅使用 v3 全量重建后的索引，Pilot v2.1 仅作历史诊断参考。
