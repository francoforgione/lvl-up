import json
from pathlib import Path

import anthropic

from lvlup.chunking import chunk_abstract
from lvlup.config import get_settings
from lvlup.ingestion.pipeline import fetch_raw_works

GROUND_TRUTH_PATH = Path("data/ground_truth.jsonl")

PROMPT_TEMPLATE = """You are helping build an evaluation set for a RAG system about digital wellness and habits.
Given the research abstract below, write {n} realistic questions a curious, non-expert user might ask that this abstract fully answers.
Return ONLY a JSON array of {n} strings, no extra text.

Abstract:
{abstract}
"""


def generate_questions(client: anthropic.Anthropic, model: str, abstract: str, n: int = 3) -> list[str]:
    message = client.messages.create(
        model=model,
        max_tokens=512,
        messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(n=n, abstract=abstract)}],
    )
    text = "".join(block.text for block in message.content if block.type == "text")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return []


def run(n_questions_per_chunk: int = 3, limit: int | None = None) -> None:
    settings = get_settings()
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    works = fetch_raw_works()
    if limit:
        works = works[:limit]

    GROUND_TRUTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    with GROUND_TRUTH_PATH.open("w") as f:
        for work in works:
            for chunk in chunk_abstract(work, settings.chunk_max_chars):
                questions = generate_questions(client, settings.anthropic_model_eval, chunk.text, n_questions_per_chunk)
                for question in questions:
                    f.write(json.dumps({"question": question, "chunk_id": chunk.chunk_id, "doc_id": chunk.doc_id}) + "\n")

    print(f"Ground truth written to {GROUND_TRUTH_PATH}")


if __name__ == "__main__":
    run()
