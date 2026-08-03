from __future__ import annotations

from prompts import build_metadata_prompt, load_allowed_topics
from config import settings


def test_metadata_prompt_includes_confusion_pair_few_shots():
    topics = load_allowed_topics(settings.allowed_topics_file)
    prompt = build_metadata_prompt("测试片段", topics)
    assert "婚礼" in prompt and "family" in prompt
    assert "枪毙" in prompt and "politics" in prompt
    assert "买牛" in prompt and "labor" in prompt
    assert "验血" in prompt and "medical" in prompt
    assert "0–2 个" in prompt
    examples_block = prompt.split("示例", 1)[1].split("当前片段", 1)[0]
    assert examples_block.count('{"characters"') == 8
    assert "topics[0]" in prompt
    assert "medical > politics" not in prompt


def test_metadata_gen_token_budget():
    assert settings.metadata_max_new_tokens >= 384
