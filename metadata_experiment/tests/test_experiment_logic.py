from __future__ import annotations

from logic import should_retrieve
from models import MetadataExperimentMethod


def test_book_questions_retrieve_for_both_methods():
    assert should_retrieve("Book") is True


def test_general_and_rewrite_do_not_retrieve():
    assert should_retrieve("General") is False
    assert should_retrieve("Rewrite") is False


def test_methods_cover_expected_names():
    names = {
        MetadataExperimentMethod("Query-Aware Top-4", False).name,
        MetadataExperimentMethod("Query-Aware + Metadata Top-4", True).name,
    }
    assert names == {"Query-Aware Top-4", "Query-Aware + Metadata Top-4"}
