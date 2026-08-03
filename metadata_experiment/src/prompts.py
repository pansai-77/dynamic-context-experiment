from __future__ import annotations

import json
from pathlib import Path

from models import TopicDefinition

_FEW_SHOT_EXAMPLES = """
示例（只学格式与分类；topics[0] 必须是片段最核心的一个 topic）：

片段：二喜娶凤霞，锣鼓喧天，村口办婚礼……
{"characters":["二喜","凤霞"],"topics":["family"],"keywords":["婚礼","出嫁","锣鼓"]}

片段：龙二被五花大绑枪毙，邻村挖好坑，土改后的处决……
{"characters":["龙二","福贵"],"topics":["politics"],"keywords":["枪毙","土改","处决"]}

片段：有庆验血后说要把血献给县长女人，医生在产房喊血呢……
{"characters":["有庆","医生"],"topics":["medical"],"keywords":["验血","抽血","医院"]}

片段：家珍抱着刚出生的有庆，又担心福贵在外赌博输钱……
{"characters":["家珍","有庆","福贵"],"topics":["family","medical"],"keywords":["生产","担心","赌博"]}

片段：福贵输光田产后，龙二成了地主，后来土地改革时被枪毙……
{"characters":["福贵","龙二"],"topics":["gambling","politics"],"keywords":["输光","地主","枪毙"]}

片段：福贵数钱后买下老牛，解开牛脚上的绳子……
{"characters":["福贵"],"topics":["labor"],"keywords":["买牛","老牛","数钱"]}

片段：家珍把米藏在胸口带回来，关上门熬粥……
{"characters":["家珍","有庆"],"topics":["livelihood"],"keywords":["藏米","熬粥","缺粮"]}

片段：连长躲在坑道里，弟兄们讨论要不要朝国军开炮……
{"characters":["连长","福贵"],"topics":["war"],"keywords":["坑道","战场","开炮"]}
""".strip()

_CONFLICT_RULES = """
双 topic 冲突消解（仅针对当前片段；topics[0] 必须是最核心事件）：
1. 若只有一个核心事件，只输出 1 个 topic；不要为了凑数添加弱相关 topic。
2. medical + family：验血/生产/医疗过程或医疗导致的后果 → medical 放第一。
3. politics + labor：土改/公社/食堂/批斗/游街 → politics 放第一。
4. family + labor：婚嫁/送养/重聚/亲子决定是核心 → family 放第一；纯生产劳动 → labor 放第一。
5. livelihood + labor：缺粮/藏米/换粮/饥饿是核心 → livelihood 放第一。
6. gambling + labor：赌局/输家产进行中 → gambling 放第一；败落后买牛干活 → labor 放第一。
7. 不得仅因出现家人姓名、时代背景或人物过往经历就添加弱相关 topic。
""".strip()

_RETRY_SUFFIX = (
    "\n\n【重试要求】上次输出无效。请只返回合法 JSON，不要 Markdown。"
    " topics 必须从受控词表中选 0–2 个，且 topics[0] 必须是最核心事件；"
    "keywords 恰好 3 个，characters 最多 4 个。"
)


def build_metadata_prompt(chunk_text: str, topics: list[TopicDefinition], *, retry: bool = False) -> str:
    topic_lines = "\n".join(
        f'- "{topic.id}" ({topic.label})：{topic.description}'
        for topic in topics
    )
    allowed_ids = ", ".join(f'"{topic.id}"' for topic in topics)
    prompt = (
        "请为下面的小说片段生成 metadata。只返回 JSON，不要 Markdown 或解释。\n"
        "格式：\n"
        '{"characters":["人物"],"topics":["topic_id"],"keywords":["词1","词2","词3"]}\n\n'
        "规则：\n"
        "1. 只根据【当前片段】判断，不要用整本书或时代背景。\n"
        f"2. 从受控词表 {allowed_ids} 中选择 0–2 个与当前片段核心事件直接相关的 topics。\n"
        "3. topics[0] 必须是最核心的一个 topic；topics[1] 仅在该片段确实有两个同等重要的核心事件时使用。\n"
        "4. keywords 必须恰好 3 个；characters 最多 4 个。\n"
        "5. 验血/抽血/生产/接生 → medical；枪毙/土改/食堂/公社 → politics；"
        "赌局/输家产 → gambling；买牛/耕田/割稻 → labor；藏米/缺粮/换粮 → livelihood。\n\n"
        f"{_CONFLICT_RULES}\n\n"
        f"可选 topic：\n{topic_lines}\n\n"
        f"{_FEW_SHOT_EXAMPLES}\n\n"
        f"当前片段：\n{chunk_text}\n"
    )
    if retry:
        prompt += _RETRY_SUFFIX
    return prompt


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


def ontology_payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
