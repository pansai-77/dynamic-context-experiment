from metadata_experiment.metrics import METHOD_A, METHOD_B, METHODS, should_retrieve
from metadata_experiment.retrieval import MetadataVectorStore
from qdrant_client.models import MatchAny


def test_methods_include_baseline_and_metadata():
    assert METHODS == (METHOD_A, METHOD_B)


def test_should_retrieve_only_for_book_questions():
    assert should_retrieve("Book")
    assert should_retrieve("book")
    assert not should_retrieve("General")
    assert not should_retrieve("Rewrite")


def test_topic_filter_uses_or_semantics():
    topic_filter = MetadataVectorStore.topic_filter(["老牛陪伴", "参军战争"])
    condition = topic_filter.must[0]
    assert condition.key == "topics"
    assert isinstance(condition.match, MatchAny)
    assert set(condition.match.any) == {"老牛陪伴", "参军战争"}
