import pytest

from metadata_experiment.classification import (
    TopicParseError,
    is_retryable_parse_error,
    parse_validated_topics,
)
from metadata_experiment.classification_prompts import (
    build_chunk_classification_prompt_for_text,
    classification_prompt_metadata,
)
from metadata_experiment.metadata_quality import audit_content_warnings, collect_content_warnings
from metadata_experiment.topics import TOPIC_BY_NAME, topic_names


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
    assert "不得自行创建事件名称" in prompt
    assert "broader parent topic" not in prompt
    assert "示例（虚构摘要" not in prompt
    assert "化验室取血" not in prompt
    assert "仅用于医院、医生、护士、抽血、献血、验血型、输血等医疗或献血场景" in prompt


def test_audit_helpers_only_emit_warnings():
    warnings = audit_content_warnings("福贵被抓壮丁去拉大炮。", ["参军战争"])
    assert isinstance(warnings, list)


def test_collect_content_warnings_deduplicates():
    warnings = collect_content_warnings("福贵被抓壮丁去拉大炮。", ["贫困生计"])
    assert warnings
    assert len(warnings) == len(set(warnings))


def test_parse_validated_topics_accepts_json():
    raw = '{"topics": ["有庆经历", "医疗献血"]}'
    topics = parse_validated_topics(raw, set(TOPIC_BY_NAME))
    assert topics == ["有庆经历", "医疗献血"]


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
    raw = '{"topics": ["家庭生活", "贫困生计", "死亡苦难"]}'
    with pytest.raises(TopicParseError, match="expected 1-2 topics"):
        parse_validated_topics(raw, set(TOPIC_BY_NAME))
