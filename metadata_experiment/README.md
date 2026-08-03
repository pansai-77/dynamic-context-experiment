# 实验二：Query-Aware Top-4 + Metadata 预过滤

## 研究问题

在实验一最佳折中方案 **Query-Aware Top-4** 上，Metadata 预过滤是否改善 Book 问题的检索相关性？它对检索时间的实际影响如何？

## 严格 A/B 设计

| 组别 | Book 问题 | General / Rewrite | Top-k |
| --- | --- | --- | --- |
| A：Query-Aware Top-4 | 全库向量检索 | 跳过检索 | 4 |
| B：Query-Aware + Metadata Top-4 | Top-2 topic 路由 → OR Filter → 子集向量检索 | 跳过检索 | 4 |

两组固定使用实验一相同的 PDF、问题集、切块参数、`BAAI/bge-small-zh-v1.5`、Qwen 模型、Prompt、生成参数和随机种子。实验一代码不修改。

## Metadata 与路由

索引集合为 `huozhe_meta`，每个 chunk 的 payload 包含：

- `text`、`chunk_id`、页码；
- `characters`；
- `topics`（固定受控词表）；
- `keywords`；
- `importance`。

构建索引时用冻结的规则词表标注 chunk。查询时不匹配关键词，而是将问题与全部 topic 描述用同一个 embedding 模型编码，选择相似度最高的两个 topic。Qdrant 使用两个 topic 的 OR 语义过滤候选 chunk，再返回 Top-4。

## 指标

| 指标 | 含义 |
| --- | --- |
| Hit@4 | Top-4 是否至少命中一个 Gold Chunk |
| MRR@4 | 第一个 Gold Chunk 排名的倒数 |
| Filter Accuracy | B 组 Top-2 路由 topic 是否命中 Gold Topics |
| Candidates Before/After | 过滤前、后的候选 chunk 数 |
| Router Time | topic 路由耗时，包含问题 embedding |
| Vector Time | Qdrant 向量查询耗时，也包含问题 embedding |
| Retrieval Time | Router Time + Vector Time |
| Score(0-3) | 人工回答质量评分 |

`gold_annotations.csv` 已预填 Gold Topics；**Gold Chunk IDs 必须在正式运行前由人工查看完整索引后标注**。Gold Chunk IDs 为空时，程序将 Hit@4 / MRR@4 留空，不会把“未标注”错误计算为 0。

`build_index.py` 同时生成 `metadata_experiment/data/index_catalog.xlsx`，可依据完整 chunk 文本标注 Gold Chunk IDs。不要只看 A/B 检索结果来选 Gold Chunk，否则会引入结果泄漏。

> B 组当前分别执行路由 embedding 和向量查询 embedding，因此 Router Time 与 Vector Time 都包含一次 query encoding。这是实际端到端实现耗时，应如实记录。若以后共享 query embedding，必须作为新的实现版本说明，不能与本次结果混合。

## 运行

先按根目录 README 配置 `.env` 并放入 `data/book/活着.pdf`。

```bash
python metadata_experiment/scripts/build_index.py
python metadata_experiment/scripts/run_experiment.py
```

`build_index.py` 会写入 `qdrant_storage_metadata/metadata_index_manifest.json`，并在实验一索引已存在时自动比对 chunk_id 集合。`run_experiment.py` 启动前会再次校验 manifest 与实验一索引一致性；不一致时将拒绝运行。

正式跑数前建议先校验 Gold 标注：

```bash
python metadata_experiment/scripts/validate_gold.py
python metadata_experiment/scripts/validate_gold.py --require-chunk-ids
```

小规模验证：

```bash
python metadata_experiment/scripts/run_experiment.py --question-ids Q01,Q11,Q16
```

输出位于 `metadata_experiment/results/<timestamp>/`：

- `detailed_results.xlsx`
- `summary_results.xlsx`
- `run_config.json`（运行配置、依赖版本与计时说明）

人工填写 Detailed Results 的 `Score(0-3)` 后，重新运行汇总可直接调用：

```bash
python -c "from pathlib import Path; from metadata_experiment.reporting import create_summary; create_summary(Path('DETAIL.xlsx'), Path('SUMMARY.xlsx'))"
```

## 解释边界

- 主分析应以 10 道 Book 题的 Hit@4、MRR@4 和人工 Score 为准。
- General / Rewrite 用于确认 Query-Aware 跳过检索的行为一致，不用于证明 Metadata 改善检索。
- Vector Time 下降不等于总检索更快；必须同时报告 Router Time 和 Retrieval Time。
- 题量较小，结果属于 proposal 前的小实验验证，不应表述为普遍结论。
