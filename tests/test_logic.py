from src.experiment import (
    CORE_METHODS,
    METHODS,
    OPTIONAL_METHODS,
    calculate_estimated_cost,
    resolve_methods,
    should_retrieve,
)

def test_query_aware_retrieves_only_book_questions():
    method = next(m for m in METHODS if m.name == "Query-Aware")
    assert should_retrieve("Book", method) is True
    assert should_retrieve("General", method) is False
    assert should_retrieve("Rewrite", method) is False

def test_no_rag_never_retrieves():
    method = next(m for m in METHODS if m.name == "No RAG")
    assert should_retrieve("Book", method) is False

def test_cost_calculation():
    assert calculate_estimated_cost(1_000_000, 500_000, 1.0, 2.0) == 2.0

def test_default_methods_exclude_optional():
    methods = resolve_methods()
    assert len(methods) == len(CORE_METHODS)
    assert all(method.name != "Query-Aware + Top-2" for method in methods)

def test_include_optional_adds_query_aware_top_2():
    methods = resolve_methods(include_optional=True)
    assert len(methods) == len(CORE_METHODS) + len(OPTIONAL_METHODS)
    assert methods[-1].name == "Query-Aware + Top-2"
