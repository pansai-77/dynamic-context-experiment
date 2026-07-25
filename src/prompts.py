from __future__ import annotations
from .models import RetrievedChunk

SYSTEM_PROMPT = """You are a precise research assistant.
Follow the user's task exactly.
For book questions, use only the supplied context.
If the context is insufficient, say so.
Do not invent facts or citations.
Keep the answer concise but complete."""

def build_prompt(question: str, question_type: str, retrieved_chunks: list[RetrievedChunk]) -> str:
    if retrieved_chunks:
        sections = []
        for index, item in enumerate(retrieved_chunks, start=1):
            sections.append(
                f"[Context {index} | page {item.chunk.page_number} | score {item.score:.4f}]\n"
                f"{item.chunk.text}"
            )
        context = "\n\n".join(sections)
        return (
            "Answer the question using only the context below. "
            "When useful, mention the relevant page number.\n\n"
            f"Question type: {question_type}\n"
            f"Question: {question}\n\n"
            f"Context:\n{context}\n\nAnswer:"
        )
    return f"Question type: {question_type}\nTask: {question}\n\nAnswer:"
