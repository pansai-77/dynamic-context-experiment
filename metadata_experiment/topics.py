from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from src.models import Chunk


FALLBACK_TOPIC = "其他/未分类"
PREFACE_TOPIC = "序言与创作背景"
NARRATOR_TOPIC = "叙述者见闻"

# Bump when topic names, descriptions, or taxonomy grouping change.
TOPIC_TAXONOMY_VERSION = "2026-08-04-v5.1"

# Plot-event topics used for chunk classification, query routing, and OR filtering.
EVENT_TOPICS: tuple[str, ...] = (
    "赌博败家",
    "徐家败落与父亲去世",
    "租田务农与求生",
    "凤霞被送走",
    "凤霞回家与聋哑",
    "凤霞婚姻",
    "凤霞生产死亡",
    "有庆上学与跑步",
    "有庆献血死亡",
    "参军战争",
    "回乡与母亲去世",
    "土地改革与龙二被枪决",
    "人民公社与大炼钢",
    "饥荒与借粮求生",
    "春生与文化大革命",
    "家珍患病离世",
    "二喜意外死亡",
    "苦根死亡",
    "老牛与晚年",
    NARRATOR_TOPIC,
)

META_TOPICS: tuple[str, ...] = (
    PREFACE_TOPIC,
    FALLBACK_TOPIC,
)

# Backward-compatible alias for audit helpers.
SPECIFIC_TOPICS = EVENT_TOPICS


@dataclass(frozen=True)
class TopicDefinition:
    name: str
    description: str
    cues: tuple[str, ...]


ALLOWED_TOPICS: tuple[TopicDefinition, ...] = (
    TopicDefinition(
        "赌博败家",
        "福贵在赌坊赌博、输钱、输光或大幅损失徐家家产",
        ("赌博", "赌钱", "赌场", "赌坊", "龙二", "家产", "田产"),
    ),
    TopicDefinition(
        "徐家败落与父亲去世",
        "赌博输光后搬离老宅、父亲病重或去世、家珍被接回或初期安顿；"
        "不含日常租田务农、不含买老牛、不含作者序言",
        ("搬出", "大宅", "爹病了", "爹死了", "出殡", "家珍回来", "输光"),
    ),
    TopicDefinition(
        "租田务农与求生",
        "败落后租种田地、务农劳动、为吃饭和维持生计奔波；不含买老牛与晚年相伴",
        ("租田", "种地", "庄稼", "农活", "借粮", "犁地"),
    ),
    TopicDefinition(
        "凤霞被送走",
        "凤霞被送给别人、离开或送走相关情节",
        ("凤霞", "送走", "送人", "嫁"),
    ),
    TopicDefinition(
        "凤霞回家与聋哑",
        "凤霞被送走后回家、成为聋哑或相关回归情节",
        ("凤霞", "哑巴", "回家", "聋哑"),
    ),
    TopicDefinition(
        "凤霞婚姻",
        "凤霞与二喜相亲、结婚、婚礼和婚后生活",
        ("凤霞", "二喜", "结婚", "婚礼", "相亲"),
    ),
    TopicDefinition(
        "凤霞生产死亡",
        "凤霞怀孕、生产、产后大出血或因此死亡",
        ("凤霞", "生孩子", "产后", "大出血", "血"),
    ),
    TopicDefinition(
        "有庆上学与跑步",
        "有庆上学、奔跑、学校生活及非献血的核心日常",
        ("有庆", "上学", "跑步", "学校"),
    ),
    TopicDefinition(
        "有庆献血死亡",
        "有庆为县长夫人献血、被过量抽血并因此死亡",
        ("有庆", "抽血", "献血", "医院", "医生", "血"),
    ),
    TopicDefinition(
        "参军战争",
        "福贵被抓壮丁、在军队中经历战争、逃亡或被俘",
        ("当兵", "壮丁", "大炮", "解放军", "国民党", "战场"),
    ),
    TopicDefinition(
        "回乡与母亲去世",
        "福贵从战场回乡、与母亲团聚或母亲去世",
        ("娘", "母亲", "回乡", "回家", "娘死了"),
    ),
    TopicDefinition(
        "土地改革与龙二被枪决",
        "土改、划成分、龙二受审被枪决、徐家田地重新分配",
        ("地主", "土改", "龙二", "枪毙", "五枪"),
    ),
    TopicDefinition(
        "人民公社与大炼钢",
        "人民公社、大食堂、炼钢、大跃进、集体劳动",
        ("公社", "大跃进", "食堂", "队长", "炼钢", "砸锅"),
    ),
    TopicDefinition(
        "饥荒与借粮求生",
        "缺粮、饥荒、讨饭、借粮、挨饿等生存危机",
        ("饿", "粮食", "借粮", "讨饭", "饥荒", "米"),
    ),
    TopicDefinition(
        "春生与文化大革命",
        "春生当县长、文革批斗、自杀或福贵劝其活下去",
        ("春生", "县长", "走资派", "文革", "自杀"),
    ),
    TopicDefinition(
        "家珍患病离世",
        "家珍患病加重、卧床、临终与死亡",
        ("家珍", "软骨病", "病", "卧床", "死"),
    ),
    TopicDefinition(
        "二喜意外死亡",
        "二喜被水泥板砸死或其他意外死亡",
        ("二喜", "水泥板", "意外", "死"),
    ),
    TopicDefinition(
        "苦根死亡",
        "苦根吃豆子撑死或其他死亡情节",
        ("苦根", "豆子", "撑死", "死"),
    ),
    TopicDefinition(
        "老牛与晚年",
        "福贵买下即将被宰的老牛、给牛取名、与牛共同劳动和晚年相伴",
        ("老牛", "买牛", "宰牛", "福贵也老了", "它也叫福贵"),
    ),
    TopicDefinition(
        NARRATOR_TOPIC,
        "叙述者“我”在乡间采集民歌、遇见福贵、听其口述人生；框形叙事层，不是作者自序",
        ("民歌", "采集", "田间", "讲述", "问路", "听他说", "我遇到"),
    ),
    TopicDefinition(
        PREFACE_TOPIC,
        "仅用于PDF版权页、目录页，或作者自序中关于创作观念、文学现实、《老黑奴》与写作缘起的内容；不用于小说正文叙事",
        ("前言", "序言", "作家", "写作", "老黑奴", "创作", "排版软件"),
    ),
    TopicDefinition(
        FALLBACK_TOPIC,
        "无实际语义的残缺文本，或与小说情节及作者序言无关的排版说明",
        (),
    ),
)

TOPICS = ALLOWED_TOPICS
TOPIC_BY_NAME = {topic.name: topic for topic in ALLOWED_TOPICS}


def normalize_topic_name(name: str) -> str:
    return name.strip()


CHARACTERS = ("福贵", "家珍", "凤霞", "有庆", "二喜", "苦根", "龙二", "春生", "队长", "老全")


def topic_names() -> list[str]:
    return [topic.name for topic in ALLOWED_TOPICS]


def routable_topics() -> tuple[TopicDefinition, ...]:
    return tuple(topic for topic in ALLOWED_TOPICS if topic.name != FALLBACK_TOPIC)


def routable_topic_names() -> list[str]:
    return [topic.name for topic in routable_topics()]


def router_topic_documents() -> list[str]:
    return [f"{topic.name}：{topic.description}" for topic in routable_topics()]


def topic_documents() -> list[str]:
    return router_topic_documents()


def format_topic_definitions(topics: tuple[str, ...]) -> str:
    lines: list[str] = []
    for name in topics:
        topic = TOPIC_BY_NAME[name]
        lines.append(f"- {topic.name}：{topic.description}")
    return "\n".join(lines)


def annotate_auxiliary_metadata(chunk: Chunk, max_keywords: int = 8) -> dict:
    """Cue rules only populate auxiliary payload fields, never chunk topics."""
    keyword_counts: Counter[str] = Counter()
    cue_hits = 0
    for topic in ALLOWED_TOPICS:
        if topic.name == FALLBACK_TOPIC:
            continue
        for cue in topic.cues:
            count = chunk.text.count(cue)
            if count:
                cue_hits += count
                keyword_counts[cue] += count

    characters = [name for name in CHARACTERS if name in chunk.text]
    keywords = [word for word, _ in keyword_counts.most_common(max_keywords)]
    severe_events = len(re.findall(r"死|枪毙|抽血|献血|饥饿|病|埋", chunk.text))
    importance = "high" if severe_events >= 3 else "medium" if cue_hits else "low"
    return {
        "characters": characters,
        "keywords": keywords,
        "importance": importance,
    }
