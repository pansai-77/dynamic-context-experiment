from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from src.models import Chunk


FALLBACK_TOPIC = "其他/未分类"
PREFACE_TOPIC = "序言与创作背景"

# Bump when topic names, descriptions, or taxonomy grouping change.
TOPIC_TAXONOMY_VERSION = "2026-08-04-v4"

SPECIFIC_TOPICS: tuple[str, ...] = (
    "赌博败家",
    "凤霞经历",
    "有庆经历",
    "参军战争",
    "土地改革",
    "人民公社",
    "医疗献血",
    "春生经历",
    "家珍病逝",
    "二喜苦根",
    "老牛陪伴",
)

BROAD_TOPICS: tuple[str, ...] = (
    "家庭生活",
    "贫困生计",
    "死亡苦难",
    "活着信念",
)

META_TOPICS: tuple[str, ...] = (
    PREFACE_TOPIC,
    FALLBACK_TOPIC,
)


@dataclass(frozen=True)
class TopicDefinition:
    name: str
    description: str
    cues: tuple[str, ...]


# Fixed before evaluation. LLM chunk classification and query routing share this list.
ALLOWED_TOPICS: tuple[TopicDefinition, ...] = (
    TopicDefinition(
        "赌博败家",
        "福贵沉迷赌博、在赌坊输钱、输光或大幅损失徐家家产",
        ("赌博", "赌钱", "赌场", "赌坊", "龙二", "家产", "田产"),
    ),
    TopicDefinition(
        "家庭生活",
        "仅用于家人日常相处、对话与共同生活；只有不存在更具体的具体主题时才使用；不含单一重大历史或死亡事件",
        ("家珍", "凤霞", "有庆", "二喜", "苦根", "爹", "娘"),
    ),
    TopicDefinition(
        "贫困生计",
        "仅当经济困难、饥饿、缺粮、为吃饭和维持生计发愁是文本核心内容时使用；核心是生存资源匮乏",
        ("穷", "粮食", "饿", "米", "讨饭", "借粮"),
    ),
    TopicDefinition(
        "凤霞经历",
        "凤霞被送人、聋哑、与二喜结婚、生育、产后大出血或死亡",
        ("凤霞", "哑巴", "二喜", "生孩子", "产后"),
    ),
    TopicDefinition(
        "有庆经历",
        "有庆奔跑、上学、被抽血、献血致死或相关核心事件",
        ("有庆", "上学", "跑步", "抽血", "献血", "血"),
    ),
    TopicDefinition(
        "参军战争",
        "福贵被抓壮丁、在国民党或解放军军队中经历战争",
        ("当兵", "壮丁", "大炮", "解放军", "国民党", "战场"),
    ),
    TopicDefinition(
        "土地改革",
        "土改、划成分、龙二受审被枪决、徐家田地充公或重新分配",
        ("地主", "土改", "龙二", "枪毙", "五枪"),
    ),
    TopicDefinition(
        "人民公社",
        "人民公社、大食堂、炼钢、大跃进、集体劳动",
        ("公社", "大跃进", "食堂", "队长", "炼钢"),
    ),
    TopicDefinition(
        "医疗献血",
        "仅用于医院、医生、护士、抽血、献血、验血型、输血等医疗或献血场景；不适用于一般疾病、卧床、虚弱、死亡或非医疗语境",
        ("医院", "医生", "大夫", "护士", "献血", "抽血"),
    ),
    TopicDefinition(
        "春生经历",
        "春生、县长身份、文革批斗、春生自杀或轻生",
        ("春生", "县长", "走资派", "自杀", "活下去"),
    ),
    TopicDefinition(
        "家珍病逝",
        "家珍患病加重、卧床、临终与死亡",
        ("家珍", "软骨病", "病", "卧床", "死"),
    ),
    TopicDefinition(
        "二喜苦根",
        "二喜被水泥板砸死、苦根吃豆子撑死等家庭悲剧",
        ("二喜", "苦根", "水泥板", "豆子"),
    ),
    TopicDefinition(
        "老牛陪伴",
        "福贵买下即将被宰的老牛、给牛取名、与牛相伴的晚年",
        ("老牛", "买牛", "宰牛", "牛", "福贵也老了"),
    ),
    TopicDefinition(
        "死亡苦难",
        "亲人死亡、埋葬、丧亲之痛等苦难；无更具体死亡主题时可用",
        ("死", "埋", "坟", "病", "苦", "哭"),
    ),
    TopicDefinition(
        "活着信念",
        "人物在苦难中仍继续生活、忍耐和生命态度的直接表达",
        ("活下去", "命", "日子", "熬"),
    ),
    TopicDefinition(
        PREFACE_TOPIC,
        "作者序言、创作缘起、写作观念、作品背景、其他作品或现实经历对《活着》创作的启发",
        ("前言", "序言", "作家", "写作", "老黑奴", "创作"),
    ),
    TopicDefinition(
        FALLBACK_TOPIC,
        "PDF版权声明、页眉页脚、排版说明、无实际语义的残缺文本，或与小说内容及创作背景无关的内容",
        (),
    ),
)

TOPICS = ALLOWED_TOPICS
TOPIC_BY_NAME = {topic.name: topic for topic in ALLOWED_TOPICS}

# Common LLM misspellings or near-synonyms mapped to canonical ALLOWED_TOPICS names.
TOPIC_ALIASES: dict[str, str] = {
    "贫穷生计": "贫困生计",
    "贫困生活": "贫困生计",
}


def normalize_topic_name(name: str) -> str:
    cleaned = name.strip()
    return TOPIC_ALIASES.get(cleaned, cleaned)


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
