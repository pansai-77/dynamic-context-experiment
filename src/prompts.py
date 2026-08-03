from __future__ import annotations
from .models import Chunk, RetrievedChunk

SYSTEM_PROMPT = """你是一名严谨的研究助手。
请严格按照用户的任务要求作答。
不要编造事实或引用。
回答应简洁、准确且完整。"""

def _format_page_label(chunk: Chunk) -> str:
    if chunk.page_start != chunk.page_end:
        return f"第 {chunk.page_start}-{chunk.page_end} 页"
    return f"第 {chunk.page_number} 页"


def build_prompt(
    question: str,
    question_type: str,
    retrieved_chunks: list[RetrievedChunk],
) -> str:
    normalized_type = question_type.strip().lower()

    if not retrieved_chunks:
        if normalized_type == "book":
            return (
                f"题型：{question_type}\n"
                f"任务：{question}\n\n"
                "请仅依据提供的小说上下文回答问题。"
                "当前未检索到任何上下文，请明确说明上下文不足，不要编造。\n\n"
                "回答："
            )
        return (
            f"题型：{question_type}\n"
            f"任务：{question}\n\n"
            "请严格按照任务要求作答。\n\n"
            "回答："
        )

    sections = []
    for index, item in enumerate(retrieved_chunks, start=1):
        sections.append(
            f"[上下文 {index} | {_format_page_label(item.chunk)}]\n"
            f"{item.chunk.text}"
        )
    context = "\n\n".join(sections)

    if normalized_type == "book":
        instruction = (
            "请仅依据提供的小说上下文回答问题。"
            "如果上下文不足，请明确说明，不要编造。"
        )
    elif normalized_type == "general":
        instruction = (
            "请回答下面的一般知识问题。"
            "提供的上下文可能与问题无关，只在确实有帮助时使用，"
            "不要因为上下文无关而拒绝回答。"
        )
    elif normalized_type == "rewrite":
        instruction = (
            "请严格完成下面的改写任务。"
            "不要从上下文中添加原句没有的信息；"
            "如果上下文与改写任务无关，请忽略它。"
        )
    else:
        instruction = "请严格按照问题要求作答。"

    return (
        f"{instruction}\n\n"
        f"题型：{question_type}\n"
        f"任务：{question}\n\n"
        f"检索上下文：\n{context}\n\n"
        "回答："
    )
