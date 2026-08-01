from src.models import Chunk, RetrievedChunk
from src.prompts import build_prompt

def _chunk(page_number: int = 12, text: str = "示例上下文") -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id="p012-c001",
            text=text,
            source_file="活着.pdf",
            page_number=page_number,
            chunk_index=1,
        ),
        score=0.8123,
    )

def test_no_retrieval_prompt_asks_to_follow_task():
    prompt = build_prompt("示例任务", "Rewrite", [])
    assert "请严格按照任务要求作答。" in prompt
    assert "任务：示例任务" in prompt
    assert "检索上下文" not in prompt

def test_book_prompt_uses_novel_context_instruction():
    prompt = build_prompt("福贵为什么买牛？", "Book", [_chunk()])
    assert "请依据提供的小说上下文回答问题。" in prompt
    assert "请仅依据下列上下文回答问题" not in prompt
    assert "检索上下文：" in prompt
    assert "相似度" not in prompt

def test_general_prompt_allows_ignoring_irrelevant_context():
    prompt = build_prompt("Top-k 是什么？", "General", [_chunk()])
    assert "一般知识问题" in prompt
    assert "只在确实有帮助时使用" in prompt
    assert "请仅依据下列上下文回答问题" not in prompt

def test_rewrite_prompt_ignores_unrelated_context():
    prompt = build_prompt("请将下面的话改写……", "Rewrite", [_chunk()])
    assert "严格完成下面的改写任务" in prompt
    assert "如果上下文与改写任务无关，请忽略它" in prompt
    assert "请仅依据下列上下文回答问题" not in prompt
    assert "相似度" not in prompt

def test_retrieved_context_header_omits_similarity_score():
    prompt = build_prompt("问题", "Book", [_chunk(page_number=7)])
    assert "[上下文 1 | 第 7 页]" in prompt
    assert "0.8123" not in prompt
