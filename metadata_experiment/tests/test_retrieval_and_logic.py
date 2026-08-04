from metadata_experiment.metrics import METHOD_A, METHOD_B, METHODS, load_gold, should_retrieve
from metadata_experiment.metrics import filter_accuracy, parse_pipe_list, ranking_metrics


def test_methods_include_baseline_and_metadata():
    assert METHODS == (METHOD_A, METHOD_B)


def test_should_retrieve_only_for_book_questions():
    assert should_retrieve("Book")
    assert should_retrieve("book")
    assert not should_retrieve("General")
    assert not should_retrieve("Rewrite")


def test_ranking_metrics_uses_first_relevant_rank():
    assert ranking_metrics(["c1", "c2", "c3"], ["c2", "c9"]) == (1.0, 0.5)
    assert ranking_metrics(["c1"], ["c9"]) == (0.0, 0.0)
    assert ranking_metrics(["c1"], []) == (None, None)


def test_filter_accuracy_requires_gold_overlap():
    assert filter_accuracy(["有庆经历", "医疗献血"], ["有庆经历"]) == 1.0
    assert filter_accuracy(["家庭生活"], ["医疗献血"]) == 0.0
    assert filter_accuracy(["家庭生活"], []) is None


def test_parse_pipe_list_handles_blank_values():
    assert parse_pipe_list("福贵 | 有庆") == ["福贵", "有庆"]
    assert parse_pipe_list(float("nan")) == []


def test_ranking_metrics_stays_blank_without_gold_chunks():
    assert ranking_metrics(["c0001", "c0002"], []) == (None, None)


def test_gold_topics_do_not_imply_hit_at_4():
    hit, mrr = ranking_metrics(["c9999"], ["c0001"])
    assert hit == 0.0
    assert mrr == 0.0
