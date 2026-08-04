import pytest

from metadata_experiment.classification import (
    ChunkClassificationError,
    ChunkTopicClassifier,
    TopicParseError,
    is_retryable_parse_error,
    parse_validated_topics,
)
from metadata_experiment.classification_prompts import (
    build_chunk_classification_prompt_for_text,
    build_single_topic_classification_prompt,
    classification_prompt_metadata,
)
from metadata_experiment.metadata_quality import audit_content_warnings, collect_content_warnings
from metadata_experiment.topics import EVENT_TOPICS, TOPIC_BY_NAME, topic_names
from src.models import Chunk


class _FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []

    def answer(self, user_prompt: str):
        self.prompts.append(user_prompt)

        class _Generation:
            def __init__(self, answer: str) -> None:
                self.answer = answer

        if not self._responses:
            raise AssertionError("no fake LLM responses left")
        return _Generation(self._responses.pop(0))


def test_prompt_metadata_lists_allowed_topics():
    metadata = classification_prompt_metadata()
    assert metadata["allowed_topic_names"] == topic_names()
    assert metadata["allowed_topic_count"] == len(topic_names())


def test_general_prompt_is_closed_set_zero_shot():
    prompt = build_chunk_classification_prompt_for_text("示例文本")
    metadata = classification_prompt_metadata()
    assert metadata["few_shot"] is False
    for value in topic_names():
        assert value in prompt
    assert "不得改写、缩写、扩展、组合或创造任何主题名称" in prompt
    assert "broader parent topic" not in prompt
    assert "示例（虚构摘要" not in prompt
    assert "化验室取血" not in prompt
    assert "有庆献血死亡" in prompt
    assert "“贫困生计”" not in prompt
    assert "“活着信念”" not in prompt


def test_event_only_taxonomy_has_no_broad_topics():
    assert len(EVENT_TOPICS) == 20
    assert "贫困生计" not in topic_names()
    assert "老牛与晚年" in topic_names()
    assert "叙述者见闻" in topic_names()


def test_audit_helpers_only_emit_warnings():
    warnings = audit_content_warnings("福贵被抓壮丁去拉大炮。", ["参军战争"])
    assert isinstance(warnings, list)


def test_collect_content_warnings_deduplicates():
    warnings = collect_content_warnings("福贵被抓壮丁去拉大炮。", ["饥荒与借粮求生"])
    assert warnings
    assert len(warnings) == len(set(warnings))


def test_parse_validated_topics_accepts_json():
    raw = '{"topics": ["有庆上学与跑步", "有庆献血死亡"]}'
    topics = parse_validated_topics(raw, set(TOPIC_BY_NAME))
    assert topics == ["有庆上学与跑步", "有庆献血死亡"]


def test_parse_validated_topics_rejects_retired_v4_topic_names():
    raw = '{"topics": ["贫困生计"]}'
    with pytest.raises(TopicParseError, match="illegal topic"):
        parse_validated_topics(raw, set(TOPIC_BY_NAME))


def test_normalize_topic_name_only_strips_whitespace():
    from metadata_experiment.topics import normalize_topic_name

    assert normalize_topic_name(" 老牛与晚年 ") == "老牛与晚年"
    assert normalize_topic_name("贫困生计") == "贫困生计"


def test_parse_validated_topics_rejects_illegal_topic():
    raw = '{"topics": ["不存在"]}'
    with pytest.raises(TopicParseError, match="illegal topic"):
        parse_validated_topics(raw, set(TOPIC_BY_NAME))


def test_illegal_topic_is_not_retryable():
    with pytest.raises(TopicParseError, match="illegal topic"):
        parse_validated_topics('{"topics": ["不存在"]}', set(TOPIC_BY_NAME))
    assert not is_retryable_parse_error(TopicParseError("illegal topic: 不存在"))


def test_json_format_error_is_retryable():
    assert is_retryable_parse_error(TopicParseError("response is not valid JSON"))


def test_parse_validated_topics_rejects_too_many_topics():
    raw = '{"topics": ["赌博败家", "租田务农与求生", "饥荒与借粮求生"]}'
    with pytest.raises(TopicParseError, match="expected 1-2 topics"):
        parse_validated_topics(raw, set(TOPIC_BY_NAME))


def test_parse_validated_topics_enforces_single_topic_when_requested():
    raw = '{"topics": ["老牛与晚年", "参军战争"]}'
    with pytest.raises(TopicParseError, match="expected 1-1 topics"):
        parse_validated_topics(raw, set(TOPIC_BY_NAME), max_topics=1)


def test_single_topic_prompt_mentions_dual_candidates():
    prompt = build_single_topic_classification_prompt(
        "示例文本",
        dual_candidates=("参军战争", "回乡与母亲去世"),
    )
    assert "只能包含 1 个主题" in prompt
    assert "参军战争" in prompt
    assert "回乡与母亲去世" in prompt


def test_classifier_accepts_single_topic_on_first_response():
    chunk = Chunk("c0001", "福贵晚年买牛。", "活着.pdf", 1, 1, 1, 1)
    classifier = ChunkTopicClassifier(
        _FakeLLM(['{"topics": ["老牛与晚年"]}']),
        allow_dual_topics=False,
    )
    result = classifier.classify(chunk, use_cache=False)
    assert result.final_topics == ["老牛与晚年"]
    assert result.prompt_kind == "general"
    assert len(classifier.llm.prompts) == 1  # type: ignore[attr-defined]


def test_classifier_retries_with_single_topic_prompt_when_dual_returned():
    chunk = Chunk("c0001", "福贵被抓壮丁后回乡。", "活着.pdf", 1, 1, 1, 1)
    llm = _FakeLLM([
        '{"topics": ["参军战争", "回乡与母亲去世"]}',
        '{"topics": ["回乡与母亲去世"]}',
    ])
    classifier = ChunkTopicClassifier(llm, allow_dual_topics=False)
    result = classifier.classify(chunk, use_cache=False)
    assert result.final_topics == ["回乡与母亲去世"]
    assert result.prompt_kind == "single_topic_retry"
    assert result.attempts == 2
    assert "只能包含 1 个主题" in llm.prompts[1]


def test_classifier_records_failure_when_single_topic_retry_still_dual():
    chunk = Chunk("c0001", "福贵被抓壮丁后回乡。", "活着.pdf", 1, 1, 1, 1)
    classifier = ChunkTopicClassifier(
        _FakeLLM([
            '{"topics": ["参军战争", "回乡与母亲去世"]}',
            '{"topics": ["参军战争", "回乡与母亲去世"]}',
        ]),
        allow_dual_topics=False,
    )
    with pytest.raises(ChunkClassificationError, match="dual topic resolution failed"):
        classifier.classify(chunk, use_cache=False)


def test_classifier_keeps_dual_topics_when_explicitly_allowed():
    chunk = Chunk("c0001", "示例文本", "活着.pdf", 1, 1, 1, 1)
    classifier = ChunkTopicClassifier(
        _FakeLLM(['{"topics": ["凤霞婚姻", "凤霞生产死亡"]}']),
        allow_dual_topics=True,
    )
    result = classifier.classify(chunk, use_cache=False)
    assert result.final_topics == ["凤霞婚姻", "凤霞生产死亡"]
    assert result.prompt_kind == "general"
