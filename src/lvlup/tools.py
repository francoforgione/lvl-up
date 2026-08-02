"""Tool definitions and dispatch for the agentic loop.

Schemas are inferred from type hints and docstrings instead of being written by
hand, so a tool's signature stays the single source of truth.
"""

import inspect
import re
import types
import typing
from collections.abc import Callable

from lvlup.config import get_settings
from lvlup.guardrails import OUT_OF_SCOPE_MESSAGE, filter_by_relevance
from lvlup.retrieval import search

_JSON_TYPES: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def _unwrap_optional(annotation: object) -> tuple[object, bool]:
    """Turn `X | None` into (X, True); leave other annotations untouched."""
    origin = typing.get_origin(annotation)
    if origin is typing.Union or origin is types.UnionType:
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0], True
    return annotation, False


def _parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """Split a docstring into its summary and its `Args:` parameter descriptions."""
    if not doc:
        return "", {}

    lines = inspect.cleandoc(doc).splitlines()
    summary_lines: list[str] = []
    params: dict[str, str] = {}
    current: str | None = None
    in_args = False

    for line in lines:
        stripped = line.strip()
        if stripped.lower() in ("args:", "arguments:", "parameters:"):
            in_args = True
            continue
        if not in_args:
            summary_lines.append(stripped)
            continue

        match = re.match(r"^(\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$", stripped)
        if match:
            current = match.group(1)
            params[current] = match.group(2).strip()
        elif current and stripped:
            params[current] = f"{params[current]} {stripped}".strip()

    return " ".join(s for s in summary_lines if s).strip(), params


def build_tool_schema(func: Callable) -> dict:
    """Infer an Anthropic tool schema from a function's signature and docstring."""
    description, param_docs = _parse_docstring(func.__doc__ or "")
    hints = typing.get_type_hints(func)
    signature = inspect.signature(func)

    properties: dict[str, dict] = {}
    required: list[str] = []

    for name, param in signature.parameters.items():
        annotation = hints.get(name, str)
        inner, optional = _unwrap_optional(annotation)
        prop: dict = {"type": _JSON_TYPES.get(inner, "string")}
        if name in param_docs:
            prop["description"] = param_docs[name]
        properties[name] = prop

        if param.default is inspect.Parameter.empty and not optional:
            required.append(name)

    return {
        "name": func.__name__,
        "description": description,
        "input_schema": {"type": "object", "properties": properties, "required": required},
    }


def format_chunks(chunks: list[dict]) -> str:
    """Render retrieved chunks as numbered excerpts for the model to read."""
    if not chunks:
        return "No matching papers found. Try different search terms."

    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[{i}] {chunk.get('title')} ({chunk.get('publication_year', 'n/d')})\n"
            f"DOI/URL: {chunk.get('doi') or chunk.get('landing_page_url') or 'n/d'}\n"
            f"{chunk.get('text')}"
        )
    return "\n\n".join(parts)


class ToolExecutor:
    """Runs tool calls and keeps the retrieved chunks around for citations.

    The model only ever sees the formatted text, but the UI and the monitoring
    tables need the structured chunks, so they are collected on the side.
    """

    def __init__(
        self,
        top_k: int | None = None,
        topic: str | None = None,
        min_relevance_score: float | None = None,
    ):
        settings = get_settings()
        self.default_top_k = top_k or settings.retrieval_top_k
        self.forced_topic = topic
        self.min_relevance_score = (
            settings.min_relevance_score if min_relevance_score is None else min_relevance_score
        )
        self.collected_chunks: list[dict] = []
        # True once a search came back with nothing above the relevance floor,
        # i.e. the user asked something the corpus can't support.
        self.guardrail_triggered = False
        self._seen_chunk_ids: set[str] = set()

    def search_papers(self, query: str, topic: str | None = None, top_k: int | None = None) -> str:
        """Search the indexed scientific papers on digital wellness and habits.

        Args:
            query: Search query in English describing the concept to look up. The
                corpus is English-only, so always translate the user's wording.
            topic: Optional topic filter. One of: screen_time_focus,
                digital_addiction, hrv, habit_formation, zone2_training,
                executive_function.
            top_k: How many excerpts to return (defaults to the configured value).
        """
        chunks = search(
            query,
            top_k=top_k or self.default_top_k,
            topic=self.forced_topic or topic,
        )

        # Gate before collecting: weak matches must reach neither the model's
        # context nor the citation list shown to the user.
        relevant = filter_by_relevance(chunks, self.min_relevance_score)
        if not relevant:
            self.guardrail_triggered = True
            return OUT_OF_SCOPE_MESSAGE

        for chunk in relevant:
            chunk_id = chunk.get("chunk_id")
            if chunk_id not in self._seen_chunk_ids:
                self._seen_chunk_ids.add(chunk_id)
                self.collected_chunks.append(chunk)
        return format_chunks(relevant)

    @property
    def tools(self) -> list[dict]:
        return [build_tool_schema(self.search_papers)]

    def run(self, name: str, tool_input: dict) -> tuple[str, bool]:
        """Execute a tool call, returning (result_text, is_error).

        Failures come back as text rather than raising so the agentic loop can
        hand them to the model and let it recover instead of aborting the turn.
        """
        handler = getattr(self, name, None)
        if handler is None or name not in {t["name"] for t in self.tools}:
            return f"Error: unknown tool '{name}'.", True
        try:
            return handler(**tool_input), False
        except Exception as exc:  # surfaced to the model as a tool error
            return f"Error running {name}: {exc}", True
