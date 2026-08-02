from __future__ import annotations

import json
from pathlib import Path

from models import TopicDefinition

_FEW_SHOT_EXAMPLES = """
示例（只学格式与分类；婚礼/打孩子/枪毙/买牛等场景不要标错）：

片段：二喜娶凤霞，锣鼓喧天，村口办婚礼，不是批斗游街……
{"characters":["二喜","凤霞"],"topics":["family"],"keywords":["婚礼","出嫁","锣鼓"]}

片段：福贵在学校打了有庆，女老师在旁边训斥，核心是父子管教……
{"characters":["福贵","有庆"],"topics":["family"],"keywords":["打孩子","上学","管教"]}

片段：家珍回娘家后又回到福贵身边，夫妻分别又重聚……
{"characters":["家珍","福贵"],"topics":["family"],"keywords":["重聚","夫妻","送别"]}

片段：龙二被五花大绑枪毙，邻村挖好坑，这是土改后的处决……
{"characters":["龙二","福贵"],"topics":["politics"],"keywords":["枪毙","枪决","土改"]}

片段：年轻人砸烂家里的锅，队长说以后吃公共食堂……
{"characters":["家珍","队长"],"topics":["politics"],"keywords":["食堂","砸锅","公社"]}

片段：有庆验血后说要把血献给县长女人，医生在产房喊血呢……
{"characters":["有庆","医生"],"topics":["medical"],"keywords":["验血","抽血","医院"]}

片段：福贵数钱后买下老牛，解开牛脚上的绳子……
{"characters":["福贵"],"topics":["labor"],"keywords":["买牛","老牛","数钱"]}

片段：连长躲在坑道里，弟兄们讨论要不要朝国军开炮……
{"characters":["连长","福贵"],"topics":["war"],"keywords":["坑道","战场","开炮"]}

片段：家珍把米藏在胸口带回来，关上门熬粥……
{"characters":["家珍","有庆"],"topics":["livelihood"],"keywords":["藏米","熬粥","缺粮"]}

片段：福贵把骰子一扔，银子滑下桌面，龙二赢得徐家财产……
{"characters":["福贵","龙二"],"topics":["gambling"],"keywords":["赌博","输钱","骰子"]}
""".strip()


def build_metadata_prompt(chunk_text: str, topics: list[TopicDefinition]) -> str:
    topic_lines = "\n".join(
        f'- "{topic.id}" ({topic.label})：{topic.description}'
        for topic in topics
    )
    allowed_ids = ", ".join(f'"{topic.id}"' for topic in topics)
    return (
        "请为下面的小说片段生成 metadata。只返回 JSON，不要 Markdown 或解释。\n"
        "格式：\n"
        '{"characters":["人物"],"topics":["topic_id"],"keywords":["词1","词2","词3"]}\n\n'
        "规则：\n"
        "1. 只根据【当前片段】判断，不要用整本书或时代背景。\n"
        f"2. topics 从 {allowed_ids} 中选 0 或 1 个；旁白、过渡可返回 []。\n"
        "3. keywords 必须恰好 3 个；characters 最多 4 个。\n"
        "4. 婚礼/出嫁/亲子管教/夫妻重聚 → family，不是 politics；"
        "枪毙/枪决/食堂/红卫兵 → politics，不是 war；"
        "验血/抽血/生产 → medical；数钱买牛/耕田 → labor，不是 gambling。\n\n"
        f"可选 topic：\n{topic_lines}\n\n"
        f"{_FEW_SHOT_EXAMPLES}\n\n"
        f"当前片段：\n{chunk_text}\n"
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
