from lvlup.evaluation import rag_eval


def _fake_answer(cost, guardrail, chunks):
    def answer(question, system_prompt=None):
        return {"cost_usd": cost, "guardrail_triggered": guardrail, "chunks": chunks, "answer": "a"}

    return answer


def test_relevant_when_answered_excludes_refused_questions(monkeypatch, tmp_path):
    # A refused question (no chunks) can still be judged RELEVANT by the judge
    # (a well-worded "outside my scope" reads as on-topic) — it must not be
    # counted as evidence the RAG pipeline is doing well.
    ground_truth = tmp_path / "ground_truth.jsonl"
    ground_truth.write_text('{"question": "q1", "chunk_id": "c1"}\n{"question": "q2", "chunk_id": "c2"}\n')
    monkeypatch.setattr(rag_eval, "GROUND_TRUTH_PATH", ground_truth)

    answers = iter(
        [
            {"cost_usd": 0.0, "guardrail_triggered": True, "chunks": [], "answer": "refused"},
            {"cost_usd": 0.0, "guardrail_triggered": False, "chunks": [{"chunk_id": "c2"}], "answer": "real"},
        ]
    )
    monkeypatch.setattr(rag_eval, "answer", lambda question, system_prompt=None: next(answers))
    monkeypatch.setattr(rag_eval, "judge", lambda *a, **k: "RELEVANT")
    monkeypatch.setattr(rag_eval.anthropic, "Anthropic", lambda **kwargs: None)
    monkeypatch.setattr(rag_eval, "log_eval_metric", lambda *a, **k: None)

    result = rag_eval.evaluate(sample_size=2, seed=0)

    assert result["relevant_when_answered"] == 1.0  # only the 1 non-refused answer counts
    assert result["refusal_rate"] == 0.5
