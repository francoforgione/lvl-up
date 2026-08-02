"""Retrieval quality: does the search find the chunk each question came from?

Compares dense (embeddings only) vs hybrid (dense + BM25, RRF-fused) so the
better approach can be picked with evidence, not a guess. Reports hit-rate@k
and MRR@k overall and per topic — per-topic matters because chunk counts are
uneven across topics (long zone-2 abstracts split into many more chunks than
short habit-formation ones), so a single aggregate is weighted towards
whichever topic happens to produce the most chunks.
"""

import json
from collections import defaultdict
from pathlib import Path

from lvlup.monitoring.db import log_eval_metric
from lvlup.retrieval import search

GROUND_TRUTH_PATH = Path("data/ground_truth.jsonl")
MODES = ("dense", "hybrid")


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


def evaluate_mode(records: list[dict], mode: str, top_k: int) -> dict:
    overall: list[tuple[bool, float]] = []
    by_topic: dict[str, list[tuple[bool, float]]] = defaultdict(list)

    for record in records:
        results = search(record["question"], top_k=top_k, mode=mode)
        ids = [r["chunk_id"] for r in results]

        if record["chunk_id"] in ids:
            outcome = (True, 1 / (ids.index(record["chunk_id"]) + 1))
        else:
            outcome = (False, 0.0)

        overall.append(outcome)
        by_topic[record.get("topic_query") or "unknown"].append(outcome)

    hit_rate, mrr = _score(overall)
    log_eval_metric("retrieval", f"hit_rate.{mode}", hit_rate)
    log_eval_metric("retrieval", f"mrr.{mode}", mrr)

    topic_metrics = {}
    for topic, outcomes in by_topic.items():
        t_hit, t_mrr = _score(outcomes)
        topic_metrics[topic] = {"hit_rate": t_hit, "mrr": t_mrr, "n": len(outcomes)}
        log_eval_metric("retrieval", f"hit_rate.{mode}.{topic}", t_hit)

    return {"hit_rate": hit_rate, "mrr": mrr, "n": len(records), "by_topic": topic_metrics}


def evaluate(top_k: int = 5) -> dict[str, dict]:
    records = load_ground_truth()
    results = {}

    for mode in MODES:
        print(f"\n--- {mode} ---")
        results[mode] = evaluate_mode(records, mode, top_k)
        print(f"Overall (n={results[mode]['n']}, k={top_k}): "
              f"hit_rate={results[mode]['hit_rate']:.3f}  mrr={results[mode]['mrr']:.3f}")
        print(f"{'topic':22} {'n':>5} {'hit_rate':>9} {'mrr':>7}")
        for topic in sorted(results[mode]["by_topic"]):
            m = results[mode]["by_topic"][topic]
            print(f"{topic:22} {m['n']:>5} {m['hit_rate']:>9.3f} {m['mrr']:>7.3f}")

    winner = max(MODES, key=lambda m: results[m]["hit_rate"])
    print(f"\nWinner (by hit_rate): {winner}")
    return results


if __name__ == "__main__":
    evaluate()
