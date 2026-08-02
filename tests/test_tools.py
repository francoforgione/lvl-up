import pytest

from lvlup.tools import ToolExecutor, build_tool_schema, format_chunks


def sample_tool(query: str, limit: int = 5, topic: str | None = None, verbose: bool = False) -> str:
    """Look something up in the corpus.

    Args:
        query: What to search for.
        limit: How many results to return.
        topic: Optional topic filter.
    """
    return "ok"


def test_schema_uses_function_name_and_summary():
    schema = build_tool_schema(sample_tool)
    assert schema["name"] == "sample_tool"
    assert schema["description"] == "Look something up in the corpus."


def test_schema_maps_python_types_to_json_types():
    props = build_tool_schema(sample_tool)["input_schema"]["properties"]
    assert props["query"]["type"] == "string"
    assert props["limit"]["type"] == "integer"
    assert props["verbose"]["type"] == "boolean"
    # `str | None` unwraps to its inner type rather than falling back to string.
    assert props["topic"]["type"] == "string"


def test_schema_required_excludes_defaults_and_optionals():
    schema = build_tool_schema(sample_tool)
    assert schema["input_schema"]["required"] == ["query"]


def test_schema_pulls_arg_descriptions_from_docstring():
    props = build_tool_schema(sample_tool)["input_schema"]["properties"]
    assert props["query"]["description"] == "What to search for."
    assert props["limit"]["description"] == "How many results to return."
    # No `Args:` entry means no description key at all.
    assert "description" not in props["verbose"]


def test_multiline_arg_description_is_joined():
    def tool(param: str) -> str:
        """Does a thing.

        Args:
            param: A description that spans
                more than one line.
        """
        return ""

    props = build_tool_schema(tool)["input_schema"]["properties"]
    assert props["param"]["description"] == "A description that spans more than one line."


def test_format_chunks_handles_empty_results():
    assert "No matching papers" in format_chunks([])


def test_format_chunks_numbers_and_cites_results():
    text = format_chunks(
        [{"title": "Paper A", "publication_year": 2020, "doi": "10.1/a", "text": "body"}]
    )
    assert "[1] Paper A (2020)" in text
    assert "10.1/a" in text
    assert "body" in text


def test_executor_reports_unknown_tool_as_error():
    executor = ToolExecutor()
    result, is_error = executor.run("no_such_tool", {})
    assert is_error
    assert "unknown tool" in result


def test_executor_converts_exceptions_into_tool_errors(monkeypatch):
    executor = ToolExecutor()
    monkeypatch.setattr(
        "lvlup.tools.search", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("qdrant down"))
    )
    result, is_error = executor.run("search_papers", {"query": "hrv"})
    assert is_error
    assert "qdrant down" in result


def test_executor_collects_chunks_and_deduplicates(monkeypatch):
    chunk = {"chunk_id": "c1", "title": "A", "text": "t", "publication_year": 2020, "score": 0.9}
    monkeypatch.setattr("lvlup.tools.search", lambda *a, **k: [chunk])

    executor = ToolExecutor()
    executor.run("search_papers", {"query": "first"})
    executor.run("search_papers", {"query": "second"})

    # The same paper found twice should be cited once.
    assert len(executor.collected_chunks) == 1


def test_executor_forced_topic_overrides_model_choice(monkeypatch):
    captured = {}

    def fake_search(query, top_k=None, topic=None, mode=None):
        captured["topic"] = topic
        return []

    monkeypatch.setattr("lvlup.tools.search", fake_search)
    ToolExecutor(topic="hrv").run("search_papers", {"query": "q", "topic": "zone2_training"})
    assert captured["topic"] == "hrv"


@pytest.mark.parametrize("tool_name", ["search_papers"])
def test_exposed_tools_have_valid_schemas(tool_name):
    schemas = {t["name"]: t for t in ToolExecutor().tools}
    assert tool_name in schemas
    schema = schemas[tool_name]
    assert schema["description"]
    assert schema["input_schema"]["properties"]["query"]["type"] == "string"
    assert "query" in schema["input_schema"]["required"]
