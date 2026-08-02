"""Retrieval quality: does the search find the chunk each question came from?

Reports hit-rate@k and MRR@k overall and per topic. The per-topic view matters
because chunk counts are uneven across topics (long zone-2 abstracts split into
many more chunks than short habit-formation ones), so a single aggregate is
weighted towards whichever topic happens to produce the most chunks.
"""

import json
from collections import defaultdict
from pathlib import Path

from lvlup.monitoring.db import log_eval_metric
from lvlup.retrieval import search

GROUND_TRUTH_PATH = Path("data/ground_truth.jsonl")


def load_ground_truth() -> list[dict]:
    with GROUND_TRUTH_PATH.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _score(records: list[tuple[bool, float]]) -> tuple[float, float]:
    """Turn (hit, reciprocal_rank) pairs into (hit_rate, mrr)."""
    if not records:
        return 0.0, 0.0
    return (
        sum(1 for hit, _ in records if hit) / len(records),
        sum(rr for _, rr in records) / len(records),
    )


def evaluate(top_k: int = 5) -> dict:
    records = load_ground_truth()
    overall: list[tuple[bool, float]] = []
    by_topic: dict[str, list[tuple[bool, float]]] = defaultdict(list)

    for i, record in enumerate(records, start=1):
        results = search(record["question"], top_k=top_k)
        ids = [r["chunk_id"] for r in results]

        if record["chunk_id"] in ids:
            outcome = (True, 1 / (ids.index(record["chunk_id"]) + 1))
        else:
            outcome = (False, 0.0)

        overall.append(outcome)
        by_topic[record.get("topic_query") or "unknown"].append(outcome)

        if i % 100 == 0:
            print(f"  {i}/{len(records)}...")

    hit_rate, mrr = _score(overall)
    log_eval_metric("retrieval", "hit_rate", hit_rate)
    log_eval_metric("retrieval", "mrr", mrr)

    print(f"\nOverall (n={len(records)}, k={top_k}): hit_rate={hit_rate:.3f}  mrr={mrr:.3f}\n")
    print(f"{'topic':22} {'n':>5} {'hit_rate':>9} {'mrr':>7}")
    topic_metrics = {}
    for topic in sorted(by_topic):
        t_hit, t_mrr = _score(by_topic[topic])
        topic_metrics[topic] = {"hit_rate": t_hit, "mrr": t_mrr, "n": len(by_topic[topic])}
        log_eval_metric("retrieval", f"hit_rate.{topic}", t_hit)
        print(f"{topic:22} {len(by_topic[topic]):>5} {t_hit:>9.3f} {t_mrr:>7.3f}")

    return {"hit_rate": hit_rate, "mrr": mrr, "n": len(records), "by_topic": topic_metrics}


if __name__ == "__main__":
    evaluate()
