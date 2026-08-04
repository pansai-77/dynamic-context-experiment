import numpy as np

from metadata_experiment.metrics import METHOD_A, METHOD_B, METHODS, should_retrieve
from metadata_experiment.retrieval import MetadataVectorStore, TopicRouter
from qdrant_client.models import MatchAny


class _FakeEmbeddingModel:
    def encode(self, texts, *, normalize_embeddings=True, show_progress_bar=False):
        if isinstance(texts, str):
            if "牛" in texts:
                return np.array([1.0, 0.0, 0.0], dtype=float)
            if "战争" in texts:
                return np.array([0.0, 1.0, 0.0], dtype=float)
            return np.array([0.5, 0.5, 0.0], dtype=float)
        return np.vstack([self.encode(text) for text in texts])


def test_methods_include_baseline_and_metadata():
    assert METHODS == (METHOD_A, METHOD_B)


def test_should_retrieve_only_for_book_questions():
    assert should_retrieve("Book")
    assert should_retrieve("book")
    assert not should_retrieve("General")
    assert not should_retrieve("Rewrite")


def test_topic_filter_uses_or_semantics():
    topic_filter = MetadataVectorStore.topic_filter(["老牛与晚年", "参军战争"])
    condition = topic_filter.must[0]
    assert condition.key == "topics"
    assert isinstance(condition.match, MatchAny)
    assert set(condition.match.any) == {"老牛与晚年", "参军战争"}


def test_router_defaults_to_top1(monkeypatch):
    monkeypatch.setattr(
        "metadata_experiment.retrieval.router_topic_documents",
        lambda: ["老牛与晚年", "参军战争"],
    )
    monkeypatch.setattr(
        "metadata_experiment.retrieval.routable_topic_names",
        lambda: ["老牛与晚年", "参军战争"],
    )
    router = TopicRouter(_FakeEmbeddingModel())
    topics, _elapsed = router.route("福贵晚年买的那头牛叫什么名字？", top_n=1)
    assert topics == ["老牛与晚年"]


def test_router_adaptive_top2_only_when_scores_are_close(monkeypatch):
    monkeypatch.setattr(
        "metadata_experiment.retrieval.router_topic_documents",
        lambda: ["老牛与晚年", "参军战争"],
    )
    monkeypatch.setattr(
        "metadata_experiment.retrieval.routable_topic_names",
        lambda: ["老牛与晚年", "参军战争"],
    )
    router = TopicRouter(_FakeEmbeddingModel())
    topics, _elapsed = router.route(
        "福贵后来经历了什么？",
        top_n=1,
        adaptive_top2=True,
        top2_score_gap=0.5,
    )
    assert topics == ["老牛与晚年", "参军战争"]
