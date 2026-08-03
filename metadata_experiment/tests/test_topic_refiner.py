from __future__ import annotations

from topic_refiner import refine_topics, refine_metadata_topics
from models import ChunkMetadata


def test_refine_overrides_gambling_when_bleeding_signals_present():
    topics = refine_topics(
        "有庆抽血被抽死了，冲进病房找医生",
        ["抽血", "县长"],
        ["gambling"],
        {"medical", "gambling", "family"},
    )
    assert topics[0] == "medical"


def test_refine_promotes_family_for_wedding_scene():
    topics = refine_topics(
        "女婿没过门就干活，家珍说凤霞好福气，结婚",
        ["结婚", "干活"],
        ["labor"],
        {"family", "labor"},
    )
    assert topics[0] == "family"


def test_refine_promotes_politics_for_commune_canteen():
    topics = refine_topics(
        "人民公社公共食堂开伙，队长说砸锅",
        ["公社", "食堂"],
        ["labor"],
        {"politics", "labor"},
    )
    assert topics[0] == "politics"


def test_refine_metadata_topics_returns_new_dataclass():
    metadata = ChunkMetadata(
        characters=["福贵"],
        topics=["labor"],
        keywords=["公社", "食堂"],
        importance=None,
        metadata_status="ok",
    )
    refined = refine_metadata_topics(
        metadata,
        "村里办人民公社，公共食堂开伙",
        {"politics", "labor"},
    )
    assert refined.topics[0] == "politics"
