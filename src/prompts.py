from __future__ import annotations
from .models import RetrievedChunk

SYSTEM_PROMPT = """你是一名严谨的研究助手。
请严格按用户要求作答。
对于 Book 类问题，只能依据提供的上下文回答。
若上下文不足以作答，请明确说明。
不要编造事实或引用。
回答应简洁且完整。"""

def build_prompt(question: str, question_type: str, retrieved_chunks: list[RetrievedChunk]) -> str:
    if retrieved_chunks:
        sections = []
        for index, item in enumerate(retrieved_chunks, start=1):
            sections.append(
                f"[上下文 {index} | 第 {item.chunk.page_number} 页 | 相似度 {item.score:.4f}]\n"
                f"{item.chunk.text}"
            )
        context = "\n\n".join(sections)
        return (
            "请仅依据下列上下文回答问题。如有必要，可注明相关页码。\n\n"
            f"题型：{question_type}\n"
            f"问题：{question}\n\n"
            f"上下文：\n{context}\n\n回答："
        )
    return f"题型：{question_type}\n任务：{question}\n\n回答："
