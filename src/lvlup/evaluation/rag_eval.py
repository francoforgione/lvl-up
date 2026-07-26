import json
import random
from pathlib import Path

import anthropic

from lvlup.config import get_settings
from lvlup.monitoring.db import log_eval_metric
from lvlup.rag import answer

GROUND_TRUTH_PATH = Path("data/ground_truth.jsonl")

JUDGE_PROMPT = """You are judging whether an AI assistant's answer is relevant to the user's question, given the question only (not the source material).
Question: {question}
Answer: {answer}

Reply with exactly one word: RELEVANT, PARTLY_RELEVANT, or NON_RELEVANT."""


def judge(client: anthropic.Anthropic, model: str, question: str, answer_text: str) -> str:
    message = client.messages.create(
        model=model,
        max_tokens=10,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(question=question, answer=answer_text)}],
    )
    return "".join(block.text for block in message.content if block.type == "text").strip()


def evaluate(sample_size: int = 30, seed: int = 42) -> dict:
    settings = get_settings()
    with GROUND_TRUTH_PATH.open() as f:
        records = [json.loads(line) for line in f if line.strip()]
    random.Random(seed).shuffle(records)
    sample = records[:sample_size]

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    counts = {"RELEVANT": 0, "PARTLY_RELEVANT": 0, "NON_RELEVANT": 0}

    for record in sample:
        result = answer(record["question"])
        verdict = judge(client, settings.anthropic_model_eval, record["question"], result["answer"])
        counts[verdict] = counts.get(verdict, 0) + 1

    total = sum(counts.values()) or 1
    ratios = {k: v / total for k, v in counts.items()}
    for key, value in ratios.items():
        log_eval_metric("rag", key.lower(), value)

    print(counts, ratios)
    return ratios


if __name__ == "__main__":
    evaluate()
