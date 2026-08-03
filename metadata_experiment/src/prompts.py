from __future__ import annotations

import json
from pathlib import Path

from models import TopicDefinition

_FEW_SHOT_EXAMPLES = """
示例（只学格式与分类；topics[0] 必须是片段最核心的一个 topic）：

片段：二喜娶凤霞，锣鼓喧天，村口办婚礼……
{"characters":["二喜","凤霞"],"topics":["family"],"keywords":["婚礼","出嫁","锣鼓"]}

片段：女婿没过门就帮家珍下地干活，邻居说凤霞好福气……
{"characters":["二喜","家珍","凤霞"],"topics":["family"],"keywords":["结婚","帮忙","福气"]}

片段：龙二被五花大绑枪毙，邻村挖好坑，土改后的处决……
{"characters":["龙二","福贵"],"topics":["politics"],"keywords":["枪毙","土改","处决"]}

片段：村里办人民公社，公共食堂开伙，大家排队打饭……
{"characters":["福贵","家珍"],"topics":["politics"],"keywords":["公社","食堂","打饭"]}

片段：有庆验血后说要把血献给县长女人，医生在产房喊血呢……
{"characters":["有庆","医生"],"topics":["medical"],"keywords":["验血","抽血","医院"]}

片段：有庆抽血过多死了，福贵冲进病房要找医生算账……
{"characters":["有庆","福贵","医生"],"topics":["medical"],"keywords":["抽血","死亡","医院"]}

片段：家珍抱着刚出生的有庆，又担心福贵在外赌博输钱……
{"characters":["家珍","有庆","福贵"],"topics":["family","medical"],"keywords":["生产","担心","赌博"]}

片段：福贵输光田产后，龙二成了地主，后来土地改革时被枪毙……
{"characters":["福贵","龙二"],"topics":["gambling","politics"],"keywords":["输光","地主","枪毙"]}

片段：福贵数钱后买下老牛，解开牛脚上的绳子……
{"characters":["福贵"],"topics":["labor"],"keywords":["买牛","老牛","数钱"]}

片段：家珍把米藏在胸口带回来，关上门熬粥……
{"characters":["家珍","有庆"],"topics":["livelihood"],"keywords":["藏米","熬粥","缺粮"]}

片段：家珍脱掉了旗袍穿上粗布，整天劳动仍笑盈盈，只要人活得高兴就不怕穷……
{"characters":["家珍","凤霞"],"topics":["family","livelihood"],"keywords":["重聚","苦日子","高兴"]}

片段：凤霞被送给别人，家珍给她换上水红衣服，有庆还不懂发生了什么……
{"characters":["凤霞","家珍","有庆"],"topics":["family"],"keywords":["送养","分离","衣服"]}

片段：连长躲在坑道里，弟兄们讨论要不要朝国军开炮……
{"characters":["连长","福贵"],"topics":["war"],"keywords":["坑道","战场","开炮"]}
""".strip()

_CONFLICT_RULES = """
主 topic 排序与冲突消解（topics 数组第一个元素最重要）：
1. topics[0] = 片段最核心的一个事件；topics[1] = 次要背景（可选）。若只有一个核心事件，只输出 1 个 topic。
2. 优先级（高→低）：medical > politics > gambling > war > family > livelihood > labor。
3. medical + family → 验血/生产/医疗死亡时 medical 放第一。
4. politics + labor → 公社/食堂/批斗/土改时 politics 放第一。
5. family + labor → 婚嫁/送养/重聚/亲子事件时 family 放第一；只有纯干农活才选 labor。
6. livelihood + labor → 缺粮/藏米/换粮时 livelihood 放第一。
7. gambling + labor → 正在赌或输家产时 gambling 放第一；败落后买牛干活只选 labor。
8. 不得仅因出现家人姓名、县长、或过去赌博经历就添加弱相关 topic。
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
        "3. topics[0] 必须是最核心的一个 topic；若只有一个核心事件，只输出 1 个 topic，"
        "不要为了凑数添加第二个弱相关 topic。\n"
        "4. keywords 必须恰好 3 个；characters 最多 4 个。\n"
        "5. 验血/抽血/生产/接生 → medical；枪毙/土改/食堂/公社/批斗 → politics；"
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
