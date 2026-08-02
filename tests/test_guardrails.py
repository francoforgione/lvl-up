from lvlup.guardrails import OUT_OF_SCOPE_MESSAGE, filter_by_relevance
from lvlup.tools import ToolExecutor


def chunk(chunk_id: str, score: float) -> dict:
    return {
        "chunk_id": chunk_id,
        "score": score,
        "text": "abstract text",
        "title": "A paper",
        "publication_year": 2020,
    }


def test_keeps_chunks_at_or_above_threshold():
    chunks = [chunk("a", 0.9), chunk("b", 0.68), chunk("c", 0.5)]
    assert [c["chunk_id"] for c in filter_by_relevance(chunks, 0.68)] == ["a", "b"]


def test_drops_everything_when_all_below_threshold():
    chunks = [chunk("a", 0.60), chunk("b", 0.55)]
    assert filter_by_relevance(chunks, 0.68) == []


def test_empty_input_stays_empty():
    assert filter_by_relevance([], 0.68) == []


def test_missing_score_is_treated_as_irrelevant():
    assert filter_by_relevance([{"chunk_id": "a"}], 0.68) == []


def _executor_with_results(monkeypatch, results: list[dict], **kwargs) -> ToolExecutor:
    executor = ToolExecutor(**kwargs)
    monkeypatch.setattr("lvlup.tools.search", lambda *a, **kw: results)
    return executor


def test_search_papers_gates_out_irrelevant_results(monkeypatch):
    executor = _executor_with_results(monkeypatch, [chunk("a", 0.55)], min_relevance_score=0.68)

    result = executor.search_papers("best pizza recipe")

    assert result == OUT_OF_SCOPE_MESSAGE
    assert executor.guardrail_triggered is True
    # Nothing weak leaks into the citation list shown to the user.
    assert executor.collected_chunks == []


def test_search_papers_keeps_relevant_results(monkeypatch):
    executor = _executor_with_results(
        monkeypatch, [chunk("a", 0.80), chunk("b", 0.50)], min_relevance_score=0.68
    )

    result = executor.search_papers("how to improve HRV")

    assert "A paper" in result
    assert executor.guardrail_triggered is False
    assert [c["chunk_id"] for c in executor.collected_chunks] == ["a"]
