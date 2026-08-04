from __future__ import annotations

from .topics import (
    ALLOWED_TOPICS,
    BROAD_TOPICS,
    FALLBACK_TOPIC,
    META_TOPICS,
    SPECIFIC_TOPICS,
    format_topic_definitions,
    topic_names,
)

CLASSIFICATION_PROMPT_VERSION = "2026-08-04-v3"


GENERIC_FEW_SHOT_EXAMPLES = """
通用示例（只返回JSON；示例文本为虚构摘要，不是本书诊断Chunk）：
- 文本：他在赌坊里连输几把，把身上的铜钱都押光了。→ {"topics": ["赌博败家"]}
- 文本：他们被强行拉走扛大炮，穿着黄颜色的军装往北方走。→ {"topics": ["参军战争"]}
- 文本：队长带着人挨家挨户砸锅，说以后都去公共食堂吃饭。→ {"topics": ["人民公社"]}
- 文本：医生验完血型后开始抽血，那孩子脸色越来越白。→ {"topics": ["医疗献血"]}
- 文本：家珍卧床多年，说话声音越来越轻，最后在床上断了气。→ {"topics": ["家珍病逝"]}
- 文本：徐家还有百亩田地，少爷穿着新衣出门，并未缺米少粮。→ {"topics": ["家庭生活"]}
"""


def format_allowed_topic_definitions() -> str:
    sections = [
        "具体主题（优先选择，用于描述单一明确事件）：",
        format_topic_definitions(SPECIFIC_TOPICS),
        "",
        "宽泛主题（仅当没有更准确的具体主题时使用）：",
        format_topic_definitions(BROAD_TOPICS),
        "",
        "序言与元数据主题：",
        format_topic_definitions(META_TOPICS),
    ]
    return "\n".join(sections)


def build_chunk_classification_prompt(chunk_text: str) -> str:
    return (
        "请阅读以下《活着》PDF中的文本块，并从允许的主题列表中选择最相关的主题。\n\n"
        "分类原则：\n"
        "- 只根据当前文本块的主要内容分类，不要猜测前后文；\n"
        "- 优先选择最能代表主要内容的具体主题；\n"
        "- 只有不存在更准确的具体主题时，才使用宽泛主题；\n"
        "- 默认返回1个主题；只有文本确实同时包含两个独立且重要的内容时，才返回2个；\n"
        "- 不要为了同时覆盖人物和事件而强行返回两个主题；\n"
        "- 只能使用给定主题，不得创建新主题；主题名称必须与列表完全一致；\n"
        "- “贫困生计”只用于经济困难、挨饿、缺粮等生存压力是核心内容的情况；\n"
        "- “医疗献血”只用于医院、医生、抽血、献血等医疗场景；\n"
        "- “赌博败家”只用于赌博、赌坊、输钱、败家等内容；\n"
        "- PDF声明、页眉页脚和排版文字应使用“其他/未分类”；\n"
        "- “其他/未分类”不能与其他任何主题同时返回；\n"
        "- 只返回JSON，不要输出解释。\n"
        f"{GENERIC_FEW_SHOT_EXAMPLES}\n"
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
    }
