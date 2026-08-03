# 实验二：Query-Aware Top-4 + Metadata 预过滤

## 研究问题

在实验一最佳折中方案 **Query-Aware Top-4** 上，加入 Metadata 预过滤后，对 Input Tokens、响应延迟和回答质量有何影响？

## 严格 A/B 设计

| 组别 | Book 问题 | General / Rewrite | Top-k |
| --- | --- | --- | --- |
| A：Query-Aware Top-4 | 全库向量检索 | 跳过检索 | 4 |
| B：Query-Aware + Metadata Top-4 | Top-2 topic 路由 → OR Filter → 子集向量检索 | 跳过检索 | 4 |

控制变量：

1. 两组 Top-k 都是 4。
2. 两组使用相同的问题集。
3. 两组使用相同的 PDF、切块结果和向量数据。
4. 两组使用相同的 Embedding 模型。
5. 两组使用相同的 Qwen 模型。
6. 两组的 Prompt、temperature、seed、max_new_tokens 保持一致。
7. 两组唯一的主要区别是 B 组增加 Metadata 预过滤。
8. General 和 Rewrite 题目沿用实验一的 Query-Aware 规则，不进行书籍检索。
9. 只有 Book 问题执行向量检索。

实验一代码不修改。

## Metadata 与路由

索引集合为 `huozhe_meta`。Chunk Metadata 与 Query Router 共用同一套 `ALLOWED_TOPICS`：

- 索引端只能写入 `ALLOWED_TOPICS` 中的 topic；
- Router 端只能从同一套 `ALLOWED_TOPICS` 中选择 topic；
- 不允许索引端和 Router 端使用不同的 topic 名称。

查询时不匹配关键词，而是将问题与全部 topic 描述用同一个 embedding 模型编码，选择相似度最高的两个 topic。Qdrant 使用两个 topic 的 OR 语义过滤候选 chunk，再返回 Top-4。

无法通过 cue 规则判断的 chunk 会标记为 `其他/未分类`。这是实现兜底，不应作为实验结论依据。

## 主要指标

实验结果需直接比较以下三项：

| 指标 | 含义 |
| --- | --- |
| Input Tokens | 送入 LLM 的输入 token 数 |
| Total Time(ms) | 端到端响应时间（检索 + 生成） |
| Score(0-3) | 人工回答质量评分 |

Detailed Results 另记录 Retrieval Time(ms)、LLM Time(ms) 等延迟拆分，便于分析 Metadata 预过滤对检索耗时的影响。

> B 组当前分别执行路由 embedding 和向量查询 embedding，因此 Router Time 与 Vector Time 都包含一次 query encoding。报告延迟时应以 Retrieval Time 和 Total Time 为准。

## 可选诊断字段

Detailed Results 可保留以下字段，供需要时参考；**不是实验指导书要求，也不要求人工填写 Gold 数据**：

- Hit@4、MRR@4：仅当 `gold_annotations.csv` 中填写了 Gold Chunk IDs 时才会计算；为空时留空，不会记为 0。
- Filter Accuracy：仅当填写了 Gold Topics 且运行 B 组时才会计算。
- Candidates Before/After、Routed Topics：用于观察 Metadata 过滤行为。

## 运行

先按根目录 README 配置 `.env` 并放入 `data/book/活着.pdf`。

```bash
python metadata_experiment/scripts/build_index.py
python metadata_experiment/scripts/run_experiment.py
```

`build_index.py` 会写入 `qdrant_storage_metadata/metadata_index_manifest.json`，并在实验一索引已存在时自动比对 chunk_id 集合。

小规模验证：

```bash
python metadata_experiment/scripts/run_experiment.py --question-ids Q01,Q11,Q16
```

输出位于 `metadata_experiment/results/<timestamp>/`：

- `detailed_results.xlsx`
- `summary_results.xlsx`
- `run_config.json`

人工填写 Detailed Results 的 `Score(0-3)` 后，可重新生成汇总：

```bash
python -c "from pathlib import Path; from metadata_experiment.reporting import create_summary; create_summary(Path('DETAIL.xlsx'), Path('SUMMARY.xlsx'))"
```

## 解释边界

- 主分析以 Input Tokens、Total Time(ms) 和人工 Score 为准，直接比较 A/B 两组。
- General / Rewrite 用于确认 Query-Aware 跳过检索的行为一致，不用于证明 Metadata 改善检索。
- 题量较小，结果属于 proposal 前的小实验验证，不应表述为普遍结论。
