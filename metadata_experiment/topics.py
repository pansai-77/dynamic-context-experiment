from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from src.models import Chunk


FALLBACK_TOPIC = "其他/未分类"


@dataclass(frozen=True)
class TopicDefinition:
    name: str
    description: str
    cues: tuple[str, ...]


# Fixed before evaluation. Index annotation and query routing must both use this list.
ALLOWED_TOPICS: tuple[TopicDefinition, ...] = (
    TopicDefinition("赌博败家", "福贵赌博、赌坊、输掉家产和徐家衰败", ("赌博", "赌钱", "赌场", "赌坊", "龙二", "家产", "田产")),
    TopicDefinition("家庭生活", "福贵、家珍与子女的家庭关系和共同生活", ("家珍", "凤霞", "有庆", "二喜", "苦根", "爹", "娘")),
    TopicDefinition("贫困生计", "贫困、劳动、种田、饥饿和维持生计", ("穷", "田", "地", "干活", "粮食", "饿", "米")),
    TopicDefinition("凤霞经历", "凤霞被送人、婚姻、生育与死亡", ("凤霞", "哑巴", "二喜", "生孩子", "产后")),
    TopicDefinition("有庆经历", "有庆上学、跑步、献血和死亡", ("有庆", "上学", "跑步", "抽血", "献血", "血")),
    TopicDefinition("参军战争", "福贵被抓壮丁、参军和战争经历", ("当兵", "壮丁", "大炮", "解放军", "国民党", "战场")),
    TopicDefinition("土地改革", "地主身份、土地改革、龙二受审和枪决", ("地主", "土改", "龙二", "枪毙", "五枪")),
    TopicDefinition("人民公社", "人民公社、大跃进、集体劳动和公共食堂", ("公社", "大跃进", "食堂", "队长", "炼钢")),
    TopicDefinition("医疗献血", "医院、医生、献血、抽血和医疗事件", ("医院", "医生", "大夫", "护士", "献血", "抽血")),
    TopicDefinition("春生经历", "春生、县长、文化大革命和轻生", ("春生", "县长", "走资派", "自杀", "活下去")),
    TopicDefinition("家珍病逝", "家珍患病、卧床以及离世", ("家珍", "软骨病", "病", "卧床", "死")),
    TopicDefinition("二喜苦根", "二喜、苦根及其家庭悲剧", ("二喜", "苦根", "水泥板", "豆子")),
    TopicDefinition("老牛陪伴", "福贵买下老牛、给牛取名以及晚年陪伴", ("老牛", "买牛", "宰牛", "牛", "福贵也老了")),
    TopicDefinition("死亡苦难", "亲人死亡、疾病、饥荒和人生苦难", ("死", "埋", "坟", "病", "苦", "哭")),
    TopicDefinition("活着信念", "面对苦难仍继续活着、忍耐和生命态度", ("活着", "活下去", "命", "日子", "熬")),
    TopicDefinition(
        FALLBACK_TOPIC,
        "无法明确归入其他受控主题的片段",
        (),
    ),
)

TOPICS = ALLOWED_TOPICS
TOPIC_BY_NAME = {topic.name: topic for topic in ALLOWED_TOPICS}
CHARACTERS = ("福贵", "家珍", "凤霞", "有庆", "二喜", "苦根", "龙二", "春生", "队长", "老全")


def topic_names() -> list[str]:
    return [topic.name for topic in ALLOWED_TOPICS]


def routable_topic_names() -> list[str]:
    return topic_names()


def topic_documents() -> list[str]:
    return [f"{topic.name}：{topic.description}" for topic in ALLOWED_TOPICS]


def annotate_chunk(chunk: Chunk, max_topics: int = 4, max_keywords: int = 8) -> dict:
    topic_scores: list[tuple[str, int]] = []
    keyword_counts: Counter[str] = Counter()
    for topic in ALLOWED_TOPICS:
        if topic.name == FALLBACK_TOPIC:
            continue
        score = 0
        for cue in topic.cues:
            count = chunk.text.count(cue)
            if count:
                score += count
                keyword_counts[cue] += count
        if score:
            topic_scores.append((topic.name, score))

    topic_scores.sort(key=lambda item: (-item[1], topic_names().index(item[0])))
    selected_topics = [name for name, _ in topic_scores[:max_topics]]
    if not selected_topics:
        selected_topics = [FALLBACK_TOPIC]

    characters = [name for name in CHARACTERS if name in chunk.text]
    keywords = [word for word, _ in keyword_counts.most_common(max_keywords)]
    severe_events = len(re.findall(r"死|枪毙|抽血|献血|饥饿|病|埋", chunk.text))
    importance = "high" if severe_events >= 3 else "medium" if topic_scores else "low"
    return {
        "characters": characters,
        "topics": selected_topics,
        "keywords": keywords,
        "importance": importance,
    }
