🌐 [Versión en español](README.md)

# Lvl Up Coach

Final project for [LLM Zoomcamp 2026](https://github.com/DataTalksClub/llm-zoomcamp) (DataTalksClub).

A RAG **digital wellness and habits** coach that answers questions like:

- "How does passive scrolling affect focus?"
- "How do I improve my HRV?"
- "Does zone 2 training help executive function?"

...citing real evidence from scientific papers indexed from [OpenAlex](https://openalex.org/).

## Problem

Information about focus, habits, digital wellness, and recovering from compulsive patterns is
scattered across unsourced blog posts, Reddit threads, and YouTube videos — mixing real advice
with anecdote. When someone looks for a specific answer ("does passive scrolling hurt attention?",
"does zone 2 training help executive function?", "what does the science say about breaking a
compulsive habit?"), they rarely find one that cites the real evidence behind it.

**Lvl Up Coach** solves this with a chat that answers digital wellness, habits, and cognitive
health questions by citing real scientific papers — not the LLM's general knowledge. It's built
for anyone who wants to understand and improve their relationship with technology, their
attention, or their habits based on evidence, without having to read academic papers themselves.
It's the final project for LLM Zoomcamp 2026, and at the same time the conversational core of a
broader cognitive coaching product (24/7 chat + habit tracking + physiological signals) being
developed outside this course.

## Capstone checklist

Direct mapping to the 5 blocks the course capstone asks for, each one with the code and the
README section where it's solved — meant as an index to navigate the repo without having to
guess where each piece lives.

| Capstone requirement | What we do | Where |
|---|---|---|
| **Searchable knowledge base** — choose a dataset, ingest, clean, and store it for retrieval | OpenAlex → dlt → Postgres (`raw`) → chunking → dense + sparse embeddings → Qdrant | [`ingestion/`](src/lvlup/ingestion/), [`chunking.py`](src/lvlup/chunking.py), [`embeddings.py`](src/lvlup/embeddings.py), [`indexing.py`](src/lvlup/indexing.py) — see [How to run it](#how-to-run-it), step 3 |
| **Retrieval pipeline** — retrieve context, assemble prompt, call LLM, grounded answers | Agentic loop: Claude decides when and what to search, answers only from retrieved excerpts and cites the source | [`rag.py`](src/lvlup/rag.py), [`tools.py`](src/lvlup/tools.py), [`retrieval.py`](src/lvlup/retrieval.py) — see [Architecture](#architecture) |
| **Evaluation process** — search metrics or LLM-as-a-Judge | hit-rate/MRR comparing dense vs hybrid + LLM-as-judge comparing baseline vs guarded prompt | [`evaluation/`](src/lvlup/evaluation/) — see [How to run it](#how-to-run-it), step 4 |
| **User-facing interface** — UI or API | Streamlit chat: answer + cited sources + cost/tokens/iterations per turn + 👍/👎 button | [`app/streamlit_app.py`](app/streamlit_app.py) — see [How to run it](#how-to-run-it), step 5 |
| **Monitoring & feedback loops** — track queries, feedback, and performance over time | Every turn is logged to Postgres (tokens, cost, guardrail, cited chunks, feedback); Grafana charts 8 panels on that data | [`monitoring/`](src/lvlup/monitoring/), [`grafana/provisioning/`](grafana/provisioning/) — see [Guardrails](#guardrails) |

## Stack

| Component | Technology |
|---|---|
| Ingestion | [dlt](https://dlthub.com/) |
| Data source | [OpenAlex API](https://docs.openalex.org/) (abstracts on screen time, digital addiction, HRV, habit formation, zone 2 training, executive function, compulsive sexual behavior/problematic pornography use, prefrontal cortex development) |
| Vector DB | [Qdrant](https://qdrant.tech/) |
| Embeddings | [FastEmbed](https://github.com/qdrant/fastembed) local (`BAAI/bge-small-en-v1.5`, CPU, no cost) |
| LLM | [Claude](https://www.anthropic.com/claude) (Anthropic API) |
| UI | [Streamlit](https://streamlit.io/) |
| Monitoring | Postgres + [Grafana](https://grafana.com/) |
| Orchestration | docker-compose |

## Architecture

```
OpenAlex API --(dlt)--> Postgres (raw schema)
                              |
                chunking + FastEmbed (dense + BM25 sparse)
                              v
                Qdrant (lvlup_papers, named vectors)
                              ^
                              | search_papers (tool)
                              |
Streamlit UI --question + history--> rag.py (agentic loop) --(Claude)--> answer + citations
      |                                                                          |
      +---- feedback (👍/👎) + tokens/cost/guardrail ------------> Postgres (app schema)
                                                                                  |
                                                                          Grafana dashboards
```

Retrieval is **not** a fixed step: `search_papers` is a tool Claude decides to invoke (once,
several times with different queries, in parallel, or not at all if prior context is enough).

Evaluation (Claude-generated ground truth, retrieval hit-rate/MRR, RAG LLM-as-judge) also writes
to this same `app.eval_runs` schema, so metrics can be charted in Grafana.

## Guardrails

Without a guardrail, retrieval always returns its nearest top-k, however far away — the LLM can
end up answering from general knowledge instead of admitting the question is outside its domain.
Two layers:

1. **System prompt** ([rag.py](src/lvlup/rag.py)): explicit scope to the corpus's 8 topics, never
   answer from general knowledge, no diagnoses or medical advice, non-judgmental tone on
   compulsive-behavior questions, refer to a professional on signs of crisis. In practice, this
   layer alone already rejects almost everything out of scope without even calling the tool.
2. **Score filter** ([guardrails.py](src/lvlup/guardrails.py)): if the model searches anyway,
   `search_papers` drops chunks below `MIN_RELEVANCE_SCORE` (`0.68`). Measured: in-domain
   0.75-0.85, out-of-domain 0.54-0.64 — the threshold sits in the gap between the two.

   Always runs in `dense` mode: `hybrid`'s RRF fusion score reflects rank position, not semantic
   similarity, so it isn't a usable relevance signal. `search_papers` searches twice — `dense` to
   decide whether the topic is supported, `hybrid` for the content that gets cited.

Verified against the 472 ground-truth questions: the filter never triggered a false rejection
(`guardrail_false_rejection_rate = 0.0`).

## Repo structure

```
src/lvlup/
├── config.py                    # settings + OpenAlex topic list
├── ingestion/                   # dlt source + pipeline (OpenAlex -> Postgres raw)
├── chunking.py                  # abstract -> chunk(s)
├── embeddings.py                # FastEmbed wrappers: dense (bge-small) + sparse (BM25)
├── indexing.py                  # Postgres raw -> Qdrant (named vectors: dense + bm25)
├── retrieval.py                 # dense or hybrid (dense + BM25, RRF fusion) search on Qdrant
├── guardrails.py                # relevance score filter (avoids hallucinating out of domain)
├── tools.py                     # search_papers tool + schema inference from docstrings
├── rag.py                       # agentic loop (function calling + memory + citations)
├── costs.py                     # token accounting and cost estimate
├── evaluation/                  # ground truth, retrieval eval, RAG eval (LLM-as-judge)
└── monitoring/                  # schema.sql + Postgres logging helpers
app/streamlit_app.py             # chat UI
grafana/provisioning/            # Postgres datasource + dashboard
scripts/run_pipeline.py          # orchestrates ingest -> index
tests/                           # unit tests (chunking, OpenAlex, retrieval, tools, agentic loop)
```

## How to run it

### 1. Install dependencies

```bash
make install          # creates .venv, installs the package in editable mode
cp .env.example .env  # fill in ANTHROPIC_API_KEY and OPENALEX_EMAIL
```

### 2. Start dev services (Qdrant + Postgres)

```bash
make dev-up
make init-db   # creates the "app" schema (conversations, messages, feedback, eval_runs)
```

### 3. Ingest and index the papers

```bash
make ingest    # dlt: OpenAlex -> Postgres (raw.openalex_works schema)
make index     # chunking + embeddings + upsert to Qdrant
```

### 4. Evaluation (retrieval + RAG)

```bash
make ground-truth     # Claude generates questions per chunk -> data/ground_truth.jsonl
make eval-retrieval   # hit-rate@k and MRR@k, dense vs hybrid
make eval-rag RAG_EVAL_ARGS="--prompt guarded"    # LLM-as-judge
make eval-rag RAG_EVAL_ARGS="--prompt baseline"   # same, with the pre-guardrails prompt
```

Aggregate results are logged to `app.eval_runs` to view in Grafana. Two approaches of each type
were evaluated, and the winner was kept as the default:

**Retrieval — dense vs hybrid** (472 questions, top-5):

| Mode | hit-rate | MRR |
|---|---|---|
| dense (embeddings only) | 0.750 | 0.574 |
| **hybrid (dense + BM25, RRF fusion)** | **0.790** | **0.608** |

`hybrid` wins and is the default in [retrieval.py](src/lvlup/retrieval.py) (`search(..., mode="hybrid")`).
Note: the guardrails' relevance gate always runs in `dense` mode — `hybrid`'s RRF fusion score
isn't a calibrated relevance signal, see [Guardrails](#guardrails).

**RAG — baseline vs guarded prompt** (30 questions, fixed sample):

| Prompt | RELEVANT (all) | Refused without evidence | RELEVANT (answered only) |
|---|---|---|---|
| baseline (no guardrails) | 53.3% | 40.0% | 88.9% |
| guarded (with guardrails) | 36.7% | 70.0% | 88.9% |

The expanded corpus brought in noise (`prefrontal_cortex_development` matches some off-topic
papers through loose keyword overlap). `guarded` refuses those cases more often than `baseline` —
hence its lower raw RELEVANT ratio, even though the judge conflates "correct refusal" with "bad
answer." The fair metric (relevance *when it does answer*) is tied at 88.9%: neither prompt
sacrifices quality when answering; the difference is how often it chooses not to. `guarded` stays
the default for that reason: we'd rather it abstain than make something up.

### 5. Try the chat

```bash
make app   # streamlit run app/streamlit_app.py
```

### 6. Full stack with docker-compose

```bash
make up   # docker compose up --build: qdrant + postgres + grafana + app
```

- App: http://localhost:8501
- Grafana: http://localhost:3000 (admin/admin)

`docker-compose.yml` reuses the same named volumes as `docker-compose.dev.yml` (same project
name), so if you already ingested/indexed via `make dev-up`, the full stack starts up with the
data already loaded — no need to repeat `make ingest`/`make index`.

### Tests

```bash
make test
```

## Notes

- Embeddings run 100% locally (FastEmbed, no API key) to keep costs down; Claude is only used for chat, ground-truth generation, and the evaluation judge, defaulting to the cheap model (Haiku).
- OpenAlex's polite pool requires `OPENALEX_EMAIL` in `.env` (improves API reliability, no API key needed).
- The corpus is English-only: Claude translates the query when invoking the tool, which avoids an extra LLM call per question.
- `requirements.txt` (what the `Dockerfile` installs) pins exact versions, not ranges, so the build is reproducible.
