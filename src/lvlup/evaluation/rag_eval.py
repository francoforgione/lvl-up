"""End-to-end RAG evaluation: an LLM judges whether the answer is relevant.

Also counts how often the relevance guardrail fired. These questions were
written *from* the corpus, so they should all be answerable — a non-trivial
count here means MIN_RELEVANCE_SCORE is set too high and is rejecting
legitimate questions.
"""

import argparse
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
    total_cost = 0.0
    guardrail_hits = 0
    refusals: list[str] = []

    for i, record in enumerate(sample, start=1):
        result = answer(record["question"])
        total_cost += result["cost_usd"]
        if result["guardrail_triggered"]:
            guardrail_hits += 1

        # No chunks means the answer used no evidence — for a question written
        # *from* the corpus that means it was refused, which the judge scores as
        # NON_RELEVANT. Tracking it separately keeps a correct refusal (off-topic
        # paper pulled in by ingestion) from looking like a bad answer.
        refused = not result["chunks"]
        if refused:
            refusals.append(record["question"])

        verdict = judge(client, settings.anthropic_model_eval, record["question"], result["answer"])
        counts[verdict] = counts.get(verdict, 0) + 1
        print(f"  [{i}/{len(sample)}] {verdict}{' (refused)' if refused else ''}")

    total = sum(counts.values()) or 1
    ratios = {k: v / total for k, v in counts.items()}
    for key, value in ratios.items():
        log_eval_metric("rag", key.lower(), value)

    guardrail_rate = guardrail_hits / total
    refusal_rate = len(refusals) / total
    answered = total - len(refusals)
    # Relevance among questions the agent actually attempted, which is what the
    # RAG pipeline is responsible for.
    relevance_when_answered = counts["RELEVANT"] / answered if answered else 0.0

    log_eval_metric("rag", "guardrail_false_rejection_rate", guardrail_rate)
    log_eval_metric("rag", "refusal_rate", refusal_rate)
    log_eval_metric("rag", "relevant_when_answered", relevance_when_answered)

    print(f"\nVerdicts: {counts}")
    print(f"Ratios:   { {k: round(v, 3) for k, v in ratios.items()} }")
    print(f"Score gate fired on {guardrail_hits}/{total} in-corpus questions ({guardrail_rate:.1%})")
    print(f"Refused without evidence: {len(refusals)}/{total} ({refusal_rate:.1%})")
    print(f"RELEVANT among the {answered} answered: {relevance_when_answered:.1%}")
    print(f"Total cost: ${total_cost:.4f}")

    if refusals:
        print("\nRefused questions (check whether ingestion pulled in off-topic papers):")
        for question in refusals:
            print(f"  - {question[:110]}")

    return {
        **ratios,
        "guardrail_false_rejection_rate": guardrail_rate,
        "refusal_rate": refusal_rate,
        "relevant_when_answered": relevance_when_answered,
        "cost_usd": total_cost,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    evaluate(sample_size=args.sample_size, seed=args.seed)


if __name__ == "__main__":
    main()
