from metadata_experiment.metrics import filter_accuracy, parse_pipe_list, ranking_metrics
from metadata_experiment.topics import annotate_chunk
from src.models import Chunk


def test_ranking_metrics_uses_first_relevant_rank():
    assert ranking_metrics(["c1", "c2", "c3"], ["c2", "c9"]) == (1.0, 0.5)
    assert ranking_metrics(["c1"], ["c9"]) == (0.0, 0.0)
    assert ranking_metrics(["c1"], []) == (None, None)


def test_filter_accuracy_requires_gold_topic_overlap():
    assert filter_accuracy(["老牛陪伴", "家庭生活"], ["老牛陪伴"]) == 1.0
    assert filter_accuracy(["家庭生活"], ["医疗献血"]) == 0.0
    assert filter_accuracy(["家庭生活"], []) is None


def test_parse_pipe_list_handles_blank_values():
    assert parse_pipe_list("老牛陪伴 | 活着信念") == ["老牛陪伴", "活着信念"]
    assert parse_pipe_list(float("nan")) == []


def test_chunk_annotation_adds_controlled_metadata():
    chunk = Chunk("c1", "有庆在医院被医生抽血。", "活着.pdf", 1, 1, 1, 1)
    metadata = annotate_chunk(chunk)
    assert "有庆经历" in metadata["topics"]
    assert "医疗献血" in metadata["topics"]
    assert "有庆" in metadata["characters"]
    assert metadata["importance"] in {"low", "medium", "high"}

