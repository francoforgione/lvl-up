"""Agentic loop tests driven by a stub client — no API calls, no Qdrant."""

from dataclasses import dataclass

import pytest

from lvlup import rag
from lvlup.costs import TokenUsage, estimate_cost


@dataclass
class Usage:
    input_tokens: int
    output_tokens: int


@dataclass
class TextBlock:
    text: str
    type: str = "text"


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict
    type: str = "tool_use"


@dataclass
class FakeResponse:
    content: list
    stop_reason: str
    usage: Usage


class FakeClient:
    """Replays a scripted list of responses and records the requests it saw."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.requests = []
        self.messages = self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if not self._responses:
            raise AssertionError("FakeClient ran out of scripted responses")
        return self._responses.pop(0)


@pytest.fixture
def patch_client(monkeypatch):
    def _patch(responses):
        client = FakeClient(responses)
        monkeypatch.setattr(rag.anthropic, "Anthropic", lambda **kwargs: client)
        return client

    return _patch


@pytest.fixture(autouse=True)
def stub_search(monkeypatch):
    monkeypatch.setattr(
        "lvlup.tools.search",
        lambda *a, **k: [
            {
                "chunk_id": "c1",
                "title": "Paper A",
                "publication_year": 2021,
                "text": "evidence",
                "score": 0.9,
            }
        ],
    )


def text_response(text, tokens=(10, 5)):
    return FakeResponse([TextBlock(text)], "end_turn", Usage(*tokens))


def tool_response(query="focus", tokens=(20, 15), block_id="t1"):
    return FakeResponse(
        [ToolUseBlock(block_id, "search_papers", {"query": query})], "tool_use", Usage(*tokens)
    )


def test_answers_directly_without_searching(patch_client):
    patch_client([text_response("Hola")])
    result = rag.answer("hola")

    assert result["answer"] == "Hola"
    assert result["iterations"] == 1
    assert result["chunks"] == []
    assert not result["hit_iteration_limit"]


def test_runs_tool_then_answers(patch_client):
    client = patch_client([tool_response(), text_response("Segun la evidencia...")])
    result = rag.answer("como afecta el scroll al foco?")

    assert result["answer"] == "Segun la evidencia..."
    assert result["iterations"] == 2
    assert len(result["chunks"]) == 1
    assert result["chunks"][0]["title"] == "Paper A"

    # Second request must carry the assistant tool_use turn and the tool_result turn.
    second = client.requests[1]["messages"]
    assert second[-2]["role"] == "assistant"
    tool_result = second[-1]["content"][0]
    assert second[-1]["role"] == "user"
    assert tool_result["type"] == "tool_result"
    assert tool_result["tool_use_id"] == "t1"
    assert tool_result["is_error"] is False


def test_tool_errors_are_fed_back_without_aborting(patch_client, monkeypatch):
    monkeypatch.setattr(
        "lvlup.tools.search", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    client = patch_client([tool_response(), text_response("No encontre evidencia.")])
    result = rag.answer("pregunta")

    assert result["answer"] == "No encontre evidencia."
    tool_result = client.requests[1]["messages"][-1]["content"][0]
    assert tool_result["is_error"] is True
    assert "boom" in tool_result["content"]


def test_iteration_limit_stops_a_runaway_loop(patch_client):
    patch_client([tool_response(block_id=f"t{i}") for i in range(10)])
    result = rag.answer("pregunta", max_iterations=3)

    assert result["hit_iteration_limit"]
    assert result["iterations"] == 3
    assert result["answer"]  # user-facing fallback text, not empty


def test_usage_accumulates_across_iterations(patch_client):
    patch_client([tool_response(tokens=(20, 15)), text_response("ok", tokens=(30, 10))])
    result = rag.answer("pregunta")

    assert result["usage"].input_tokens == 50
    assert result["usage"].output_tokens == 25
    assert result["usage"].total_tokens == 75
    assert result["cost_usd"] > 0


def test_history_is_prepended_so_the_model_has_memory(patch_client):
    client = patch_client([text_response("Sobre HRV...")])
    history = [
        {"role": "user", "content": "hablemos de zona 2"},
        {"role": "assistant", "content": "La zona 2 es..."},
    ]
    rag.answer("y para el HRV?", history=history)

    sent = client.requests[0]["messages"]
    assert len(sent) == 3
    assert sent[0]["content"] == "hablemos de zona 2"
    assert sent[-1] == {"role": "user", "content": "y para el HRV?"}


def test_history_is_not_mutated(patch_client):
    patch_client([text_response("ok")])
    history = [{"role": "user", "content": "previa"}]
    rag.answer("nueva", history=history)
    assert len(history) == 1


def test_max_tokens_stop_reason_ends_the_turn(patch_client):
    patch_client([FakeResponse([TextBlock("truncado")], "max_tokens", Usage(10, 2048))])
    result = rag.answer("pregunta")

    assert result["stop_reason"] == "max_tokens"
    assert result["answer"] == "truncado"
    assert result["iterations"] == 1


def test_tools_are_sent_on_every_request(patch_client):
    client = patch_client([tool_response(), text_response("ok")])
    rag.answer("pregunta")

    for request in client.requests:
        assert [t["name"] for t in request["tools"]] == ["search_papers"]
        assert request["system"] == rag.SYSTEM_PROMPT


def test_cost_estimation_matches_haiku_pricing():
    usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert estimate_cost("claude-haiku-4-5-20251001", usage) == pytest.approx(6.00)


def test_cost_estimation_is_zero_for_unknown_model():
    assert estimate_cost("some-other-model", TokenUsage(1000, 1000)) == 0.0
