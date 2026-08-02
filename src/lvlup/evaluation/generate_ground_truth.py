"""Build the retrieval evaluation set: LLM-generated questions per chunk.

Each record pairs a question with the chunk it was written from, which is the
chunk retrieval is expected to find. Sampling is stratified by topic — the raw
table is ordered by ingestion, so taking the first N works would draw the whole
eval set from a single topic.
"""

import argparse
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

from lvlup.chunking import Chunk, chunk_abstract
from lvlup.config import get_settings
from lvlup.ingestion.pipeline import fetch_raw_works

GROUND_TRUTH_PATH = Path("data/ground_truth.jsonl")

PROMPT_TEMPLATE = """You are helping build an evaluation set for a RAG system about digital wellness and habits.
Given the research abstract below, write {n} realistic questions a curious, non-expert user might ask that this abstract fully answers.

Abstract:
{abstract}
"""

# Structured outputs guarantee parseable JSON. Asking for a bare array in the
# prompt gets it wrapped in markdown code fences, which json.loads rejects.
QUESTIONS_SCHEMA = {
    "type": "object",
    "properties": {"questions": {"type": "array", "items": {"type": "string"}}},
    "required": ["questions"],
    "additionalProperties": False,
}


def sample_works_by_topic(works: list[dict], n_works: int, seed: int) -> list[dict]:
    """Draw an even number of works from each topic, deterministically."""
    by_topic: dict[str, list[dict]] = defaultdict(list)
    for work in works:
        by_topic[work.get("topic_query")].append(work)

    rng = random.Random(seed)
    per_topic = max(1, n_works // len(by_topic)) if by_topic else 0

    sampled: list[dict] = []
    for topic in sorted(by_topic):
        bucket = sorted(by_topic[topic], key=lambda w: w["id"])
        sampled.extend(rng.sample(bucket, min(per_topic, len(bucket))))
    return sampled


def generate_questions(client: anthropic.Anthropic, model: str, abstract: str, n: int = 3) -> list[str]:
    message = client.messages.create(
        model=model,
        max_tokens=512,
        output_config={"format": {"type": "json_schema", "schema": QUESTIONS_SCHEMA}},
        messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(n=n, abstract=abstract)}],
    )
    text = "".join(block.text for block in message.content if block.type == "text")
    try:
        questions = json.loads(text)["questions"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return []
    return [q for q in questions if isinstance(q, str)]


def run(
    n_works: int = 60,
    n_questions_per_chunk: int = 2,
    seed: int = 42,
    workers: int = 8,
) -> Path:
    settings = get_settings()
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    works = sample_works_by_topic(fetch_raw_works(), n_works, seed)
    chunks: list[Chunk] = []
    for work in works:
        chunks.extend(chunk_abstract(work, settings.chunk_max_chars))

    print(f"Generating questions for {len(chunks)} chunks from {len(works)} works...")

    def questions_for(chunk: Chunk) -> tuple[Chunk, list[str]]:
        return chunk, generate_questions(
            client, settings.anthropic_model_eval, chunk.text, n_questions_per_chunk
        )

    # The calls are independent, so run them in parallel — sequentially this is
    # a couple of seconds per chunk.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(questions_for, chunks))

    GROUND_TRUTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with GROUND_TRUTH_PATH.open("w") as f:
        for chunk, questions in results:
            for question in questions:
                f.write(
                    json.dumps(
                        {
                            "question": question,
                            "chunk_id": chunk.chunk_id,
                            "doc_id": chunk.doc_id,
                            "topic_query": chunk.metadata.get("topic_query"),
                        }
                    )
                    + "\n"
                )
                written += 1

    print(f"Wrote {written} questions to {GROUND_TRUTH_PATH}")
    return GROUND_TRUTH_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--works", type=int, default=60, help="works to sample across all topics")
    parser.add_argument("--questions-per-chunk", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    run(
        n_works=args.works,
        n_questions_per_chunk=args.questions_per_chunk,
        seed=args.seed,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
