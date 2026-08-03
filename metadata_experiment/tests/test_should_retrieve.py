from metadata_experiment.logic import METHODS, should_retrieve


def test_should_retrieve_only_for_book_questions():
    assert should_retrieve("Book") is True
    assert should_retrieve("book") is True
    assert should_retrieve("General") is False
    assert should_retrieve("Rewrite") is False


def test_experiment_methods_are_fixed_ab_pair():
    assert METHODS == (
        "Query-Aware Top-4",
        "Query-Aware + Metadata Top-4",
    )
