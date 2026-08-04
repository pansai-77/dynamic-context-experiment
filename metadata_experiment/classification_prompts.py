from __future__ import annotations

from .topics import (
    ALLOWED_TOPICS,
    EVENT_TOPICS,
    META_TOPICS,
    format_topic_definitions,
    topic_names,
)

CLASSIFICATION_PROMPT_VERSION = "2026-08-04-v5.1.1"


def format_allowed_topic_definitions() -> str:
    sections = [
        "情节事件：",
        format_topic_definitions(EVENT_TOPICS),
        "",
        "元数据：",
        format_topic_definitions(META_TOPICS),
    ]
    return "\n".join(sections)


def format_allowed_topic_names() -> str:
    return "、".join(f"“{name}”" for name in topic_names())


def build_chunk_classification_prompt(chunk_text: str) -> str:
    return (
        "你正在执行封闭集合文本分类。请判断文本块的主要内容，"
        "从允许的情节事件或元数据主题中选择 1-2 个，名称必须逐字复制。\n\n"
        "必须遵守：\n"
        "1. topics 中的每个值必须来自允许主题名称列表；\n"
        "2. 不得改写、缩写、扩展、组合或创造任何主题名称；\n"
        "3. 默认只返回 1 个主题；仅当 chunk 同时包含两个彼此独立、"
        "都占明显篇幅的情节事件时，才返回 2 个；\n"
        "4. 只根据当前文本块分类，不推测前后文；\n"
        "5. 只输出合法 JSON，不输出解释。\n\n"
        "判断顺序：\n"
        "1. PDF 版权页、目录、页眉页脚、乱码 → “其他/未分类”；\n"
        "2. 作者自序、创作观念、《老黑奴》、文学现实等创作背景 → “序言与创作背景”；"
        "小说正文叙事（含福贵、家珍、有庆等人物故事）一律不得选此项；\n"
        "3. 否则选择最能概括本 chunk 主线的一个情节事件；\n"
        "4. “叙述者见闻”仅用于叙述者“我”在乡间遇见福贵、听其讲述的框形叙事；"
        "福贵口述的人生故事不得选此项；作者自序仍选“序言与创作背景”；\n"
        "5. “徐家败落与父亲去世”不含日常租田、买老牛；租田选“租田务农与求生”，买牛选“老牛与晚年”；\n"
        "6. “其他/未分类”不能与任何其他主题同时返回。\n\n"
        "已删除、不得使用的旧标签：家庭生活、贫困生计、死亡苦难、活着信念、"
        "医疗献血、凤霞经历、有庆经历、老牛陪伴 等 v4 名称。\n\n"
        f"允许主题名称（共 {len(topic_names())} 个）：\n"
        f"{format_allowed_topic_names()}\n\n"
        "主题定义：\n"
        f"{format_allowed_topic_definitions()}\n\n"
        "待分类文本：\n"
        f"{chunk_text}\n\n"
        '返回格式示例：{"topics": ["赌博败家"]}\n'
        '跨事件边界时最多两个：{"topics": ["凤霞婚姻", "凤霞生产死亡"]}'
    )


def build_chunk_classification_prompt_for_text(chunk_text: str) -> str:
    return build_chunk_classification_prompt(chunk_text)


def build_single_topic_classification_prompt(
    chunk_text: str,
    *,
    dual_candidates: tuple[str, str] | None = None,
) -> str:
    candidate_note = ""
    if dual_candidates is not None:
        candidate_note = (
            f"\n上一次你返回了两个主题：{dual_candidates[0]} 与 {dual_candidates[1]}。"
            "现在必须只保留最能概括本 chunk 主线的一个。\n"
        )
    return (
        "你正在执行封闭集合文本分类。请判断文本块的主要内容，"
        "从允许的情节事件或元数据主题中选择且仅选择 1 个主题，名称必须逐字复制。\n\n"
        "必须遵守：\n"
        "1. topics 数组必须且只能包含 1 个主题名称；\n"
        "2. 每个值必须来自允许主题名称列表；\n"
        "3. 不得改写、缩写、扩展、组合或创造任何主题名称；\n"
        "4. 只根据当前文本块分类，不推测前后文；\n"
        "5. 只输出合法 JSON，不输出解释。\n"
        f"{candidate_note}\n"
        "判断顺序：\n"
        "1. PDF 版权页、目录、页眉页脚、乱码 → “其他/未分类”；\n"
        "2. 作者自序、创作观念、《老黑奴》、文学现实等创作背景 → “序言与创作背景”；\n"
        "3. 叙述者“我”在乡间遇见福贵、听其讲述 → “叙述者见闻”；\n"
        "4. 否则选择最能概括本 chunk 主线的一个情节事件。\n\n"
        f"允许主题名称（共 {len(topic_names())} 个）：\n"
        f"{format_allowed_topic_names()}\n\n"
        "主题定义：\n"
        f"{format_allowed_topic_definitions()}\n\n"
        "待分类文本：\n"
        f"{chunk_text}\n\n"
        '返回格式示例：{"topics": ["赌博败家"]}'
    )


def classification_prompt_metadata() -> dict[str, object]:
    return {
        "prompt_version": CLASSIFICATION_PROMPT_VERSION,
        "allowed_topic_names": topic_names(),
        "allowed_topic_count": len(ALLOWED_TOPICS),
        "topic_definitions": format_allowed_topic_definitions(),
        "prompt_text_template": build_chunk_classification_prompt("{chunk_text}"),
        "few_shot": False,
    }
