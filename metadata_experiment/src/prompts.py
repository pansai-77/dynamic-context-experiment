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
        '  "topic_evidence": ["该 topic 在当前片段中的直接证据"],\n'
        '  "keywords": ["关键词1", "关键词2"],\n'
        '  "importance": 1\n'
        "}\n\n"
        "要求：\n"
        "1. 只根据【当前片段】判断，不得依据《活着》整本书的总体主题、时代背景或你的背景知识。\n"
        f"2. topics 只能从以下 id 中选择，允许 0–2 个：{allowed_ids}\n"
        "3. 若当前片段没有足够明确的 topic，topics 返回 []，不要为了凑数选择弱相关 topic。\n"
        "4. 每选择一个 topic，必须在 topic_evidence 中给出该片段里的直接文本证据；"
        "topic_evidence 仅用于生成校验，不会入库。\n"
        "5. 禁止使用「全书背景、时代动荡、整体苦难、小说很悲剧」作为选 topic 的理由。\n"
        "6. death_loss 仅用于人物死亡、临终、丧葬、报丧、明确哀悼，或与亲人离世直接相关的失去；"
        "不得用于失去财产、身份、工作、机会等一般性失去；"
        "不得因提到死者、背景苦难或全书悲剧基调而使用。\n"
        "7. war_conscription 仅用于抓壮丁、军队、战场、行军、战斗、战时逃难等战争核心事件；"
        "不得因出现“逃”“跑”“苦”或故事发生在战乱年代而使用；"
        "不得作为 death_loss 或其他 topic 收紧后的默认兜底标签。\n"
        "8. family_decline_gambling 仅用于赌博、输掉家产、徐家败落等直接事件；"
        "不得因“命运转折”把买牛、婚姻、死亡、土改等事件归入此类。\n"
        "9. land_reform_politics 仅用于土改、成分、批斗、游街、文革、红卫兵等政治运动核心事件；"
        "不得因时代背景本身或一般家庭变故而使用。\n"
        "10. suffering_survival 用于晚年或持续困境中的生存状态、忍耐与人生感悟；"
        "不得用于年轻时游荡、闲逛、享乐或无所事事；"
        "可在 framing 叙述（如老人与牛）中使用，但不得作为泛化兜底。\n"
        "11. marriage_family / parent_child / disease_medical / rural_labor / poverty_livelihood "
        "仅在本片段核心事件确实属于对应关系时使用；优先选择最具体、最直接的一个 topic。\n"
        "12. characters 列出片段中出现的主要人物，最多 4 个。\n"
        "13. keywords 列出 2-5 个关键词。\n"
        "14. importance 为 1-5 的整数。\n\n"
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
