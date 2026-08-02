from __future__ import annotations

import json
from pathlib import Path

from models import TopicDefinition


def build_metadata_prompt(chunk_text: str, topics: list[TopicDefinition]) -> str:
    topic_lines = "\n".join(
        f'- id="{topic.id}", label="{topic.label}", description="{topic.description}"'
        for topic in topics
    )
    allowed_ids = ", ".join(f'"{topic.id}"' for topic in topics)
    return (
        "请为下面的小说文本片段生成结构化 metadata。\n"
        "只返回 JSON，不要输出任何解释或 Markdown。\n"
        "JSON 格式：\n"
        "{\n"
        '  "characters": ["人物1", "人物2"],\n'
        '  "topics": ["topic_id"],\n'
        '  "keywords": ["关键词1", "关键词2"],\n'
        '  "importance": 1\n'
        "}\n\n"
        "要求：\n"
        f"1. topics 只能从以下 id 中选择，最多 2 个：{allowed_ids}\n"
        "2. characters 列出片段中出现的主要人物，最多 4 个\n"
        "3. keywords 列出 2-5 个关键词\n"
        "4. importance 为 1-5 的整数\n\n"
        f"可选 topic 列表：\n{topic_lines}\n\n"
        f"文本片段：\n{chunk_text}\n"
    )


def load_allowed_topics(path: Path) -> list[TopicDefinition]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        TopicDefinition(
            id=str(item["id"]),
            label=str(item["label"]),
            description=str(item["description"]),
        )
        for item in payload["topics"]
    ]
