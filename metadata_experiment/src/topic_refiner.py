from __future__ import annotations

from metadata_parsing import MAX_TOPICS, TOPIC_PRIORITY, order_topics_by_priority
from models import ChunkMetadata

# Lexical cues grouped by topic. Matched against chunk text and model keywords.
TOPIC_SIGNALS: dict[str, tuple[str, ...]] = {
    "medical": (
        "验血",
        "抽血",
        "献血",
        "医院",
        "生产",
        "接生",
        "分娩",
        "产房",
        "头晕",
        "看病",
        "大夫",
        "被抽死",
    ),
    "politics": (
        "土改",
        "土地改革",
        "枪毙",
        "批斗",
        "游街",
        "公社",
        "食堂",
        "砸锅",
        "地主",
        "红卫兵",
        "大字报",
        "煮钢铁",
        "分地",
        "处决",
    ),
    "gambling": (
        "赌坊",
        "赌博",
        "输光",
        "押注",
        "骰子",
        "沈先生",
        "赌局",
        "赌注",
    ),
    "war": (
        "战场",
        "炮火",
        "坑道",
        "伤兵",
        "连长",
        "团长",
        "逃兵",
        "军队",
        "开炮",
        "炮弹",
        "伤号",
    ),
    "family": (
        "结婚",
        "婚礼",
        "出嫁",
        "相亲",
        "嫁给我",
        "送养",
        "分离",
        "重聚",
        "丧事",
        "棺材",
        "葬礼",
        "女婿",
        "没过门",
    ),
    "livelihood": (
        "缺粮",
        "藏米",
        "熬粥",
        "换粮",
        "卖羊",
        "饥饿",
        "饿",
        "野菜",
        "借债",
        "地瓜",
        "咕咚",
    ),
    "labor": (
        "买牛",
        "耕田",
        "插秧",
        "割稻",
        "割草",
        "老牛",
        "农具",
        "棉花",
        "干活",
        "下地",
        "镰刀",
        "锄头",
    ),
}

_OVERRIDE_MARGIN = 2


def score_topic_signals(text: str, keywords: list[str]) -> dict[str, int]:
    haystack = f"{text} {' '.join(keywords)}"
    scores: dict[str, int] = {}
    for topic_id, signals in TOPIC_SIGNALS.items():
        score = 0
        for signal in signals:
            if signal in haystack:
                score += haystack.count(signal)
        scores[topic_id] = score
    return scores


def refine_topics(
    text: str,
    keywords: list[str],
    llm_topics: list[str],
    allowed_topic_ids: set[str],
) -> list[str]:
    if not llm_topics and not text.strip():
        return []

    scores = score_topic_signals(text, keywords)
    for topic_id in llm_topics:
        if topic_id in allowed_topic_ids:
            scores[topic_id] = scores.get(topic_id, 0) + 1

    ranked = sorted(
        allowed_topic_ids,
        key=lambda topic_id: (-scores.get(topic_id, 0), TOPIC_PRIORITY.get(topic_id, 99)),
    )
    signal_positive = [topic_id for topic_id in ranked if scores.get(topic_id, 0) > 0]

    if not signal_positive:
        return order_topics_by_priority(llm_topics)[:MAX_TOPICS]

    signal_primary = signal_positive[0]
    llm_primary = llm_topics[0] if llm_topics else None

    if llm_primary is None:
        return signal_positive[:MAX_TOPICS]

    if signal_primary != llm_primary:
        signal_score = scores.get(signal_primary, 0)
        llm_score = scores.get(llm_primary, 0)
        if signal_score >= llm_score + _OVERRIDE_MARGIN:
            merged = [signal_primary]
            for topic_id in signal_positive[1:] + order_topics_by_priority(llm_topics):
                if topic_id not in merged and len(merged) < MAX_TOPICS:
                    merged.append(topic_id)
            return order_topics_by_priority(merged)[:MAX_TOPICS]

    merged: list[str] = []
    for topic_id in order_topics_by_priority(llm_topics) + signal_positive:
        if topic_id not in merged and len(merged) < MAX_TOPICS:
            merged.append(topic_id)
    return order_topics_by_priority(merged)[:MAX_TOPICS]


def refine_metadata_topics(
    metadata: ChunkMetadata,
    chunk_text: str,
    allowed_topic_ids: set[str],
) -> ChunkMetadata:
    refined = refine_topics(chunk_text, metadata.keywords, metadata.topics, allowed_topic_ids)
    if refined == metadata.topics:
        return metadata
    return ChunkMetadata(
        characters=metadata.characters,
        topics=refined,
        keywords=metadata.keywords,
        importance=metadata.importance,
        metadata_status=metadata.metadata_status,
    )
