from __future__ import annotations

from .topics import (
    ALLOWED_TOPICS,
    BROAD_TOPICS,
    META_TOPICS,
    SPECIFIC_TOPICS,
    format_topic_definitions,
    topic_names,
)

CLASSIFICATION_PROMPT_VERSION = "2026-08-04-v3.3-zero-shot"


def format_allowed_topic_definitions() -> str:
    sections = [
        "具体主题：",
        format_topic_definitions(SPECIFIC_TOPICS),
        "",
        "宽泛主题：",
        format_topic_definitions(BROAD_TOPICS),
        "",
        "序言与元数据主题：",
        format_topic_definitions(META_TOPICS),
    ]
    return "\n".join(sections)


def build_chunk_classification_prompt(chunk_text: str) -> str:
    return (
        "请阅读以下文本块，从允许的主题列表中选择最相关的主题。\n\n"
        "分类原则：\n"
        "- 只根据当前文本块的主要内容分类，不要猜测前后文；\n"
        "- 默认返回 1 个主题；只有文本确实同时包含两个独立且重要的内容时，才返回 2 个；\n"
        "- 优先选择最能代表主要内容的具体主题；只有没有更准确的具体主题时才使用宽泛主题；\n"
        "- 如果具体主题已经能够覆盖主要内容，不要再同时返回语义重叠的宽泛主题；\n"
        "- 只能从允许列表中选择；topics 中的每个值必须逐字复制列表中的某一个主题；\n"
        "- 不得自行创建事件名称、人物行为描述或概括性短语；只能使用允许列表中的正式主题名称；\n"
        "- “其他/未分类”不能与其他主题同时返回；\n"
        "- 只返回合法 JSON，不要输出解释。\n\n"
        f"允许主题（共 {len(topic_names())} 个）：\n{format_allowed_topic_definitions()}\n\n"
        f"文本：\n{chunk_text}\n\n"
        '返回格式：\n{"topics": ["主题1"]}'
    )


def build_chunk_classification_prompt_for_text(chunk_text: str) -> str:
    return build_chunk_classification_prompt(chunk_text)


def classification_prompt_metadata() -> dict[str, object]:
    return {
        "prompt_version": CLASSIFICATION_PROMPT_VERSION,
        "allowed_topic_names": topic_names(),
        "allowed_topic_count": len(ALLOWED_TOPICS),
        "topic_definitions": format_allowed_topic_definitions(),
        "prompt_text_template": build_chunk_classification_prompt("{chunk_text}"),
        "few_shot": False,
    }
