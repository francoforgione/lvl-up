import json
from pathlib import Path

from lvlup.monitoring.db import log_eval_metric
from lvlup.retrieval import search

GROUND_TRUTH_PATH = Path("data/ground_truth.jsonl")


def load_ground_truth() -> list[dict]:
    with GROUND_TRUTH_PATH.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate(top_k: int = 5) -> dict:
    records = load_ground_truth()
    hits = 0
    reciprocal_ranks = []

    for record in records:
        results = search(record["question"], top_k=top_k)
        ids = [r["chunk_id"] for r in results]
        if record["chunk_id"] in ids:
            hits += 1
            reciprocal_ranks.append(1 / (ids.index(record["chunk_id"]) + 1))
        else:
            reciprocal_ranks.append(0)

    hit_rate = hits / len(records) if records else 0
    mrr = sum(reciprocal_ranks) / len(records) if records else 0
    metrics = {"hit_rate": hit_rate, "mrr": mrr, "n": len(records)}

    log_eval_metric("retrieval", "hit_rate", hit_rate)
    log_eval_metric("retrieval", "mrr", mrr)

    print(metrics)
    return metrics


if __name__ == "__main__":
    evaluate()
