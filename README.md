# Lvl Up Coach

Proyecto final de [LLM Zoomcamp 2026](https://github.com/DataTalksClub/llm-zoomcamp) (DataTalksClub).

Un coach RAG de **bienestar digital y hábitos** que responde preguntas como:

- "¿Cómo afecta el scroll pasivo al foco?"
- "¿Cómo mejoro mi HRV?"
- "¿Sirve la zona 2 para la función ejecutiva?"

...citando evidencia real de papers científicos indexados desde [OpenAlex](https://openalex.org/).

## Stack

| Componente | Tecnología |
|---|---|
| Ingesta | [dlt](https://dlthub.com/) |
| Fuente de datos | [OpenAlex API](https://docs.openalex.org/) (abstracts sobre screen time, digital addiction, HRV, habit formation, zone 2 training, executive function) |
| Vector DB | [Qdrant](https://qdrant.tech/) |
| Embeddings | [FastEmbed](https://github.com/qdrant/fastembed) local (`BAAI/bge-small-en-v1.5`, CPU, sin costo) |
| LLM | [Claude](https://www.anthropic.com/claude) (Anthropic API) |
| UI | [Streamlit](https://streamlit.io/) |
| Monitoreo | Postgres + [Grafana](https://grafana.com/) |
| Orquestación | docker-compose |

## Arquitectura

```
OpenAlex API --(dlt)--> Postgres (schema raw)
                              |
                     chunking + FastEmbed
                              v
                       Qdrant (lvlup_papers)
                              ^
                              | dense search
Streamlit UI --question--> retrieval.py --> rag.py --(Claude)--> respuesta + citas
      |                                                              |
      +---- feedback (👍/👎) ---------------------------> Postgres (schema app)
                                                                      |
                                                              Grafana dashboards
```

La evaluación (ground truth generado por Claude, hit-rate/MRR de retrieval, LLM-as-judge de RAG) también corre contra este mismo esquema `app.eval_runs`, para que las métricas se puedan graficar en Grafana.

## Estructura del repo

```
src/lvlup/
├── config.py                    # settings + lista de temas de OpenAlex
├── ingestion/                   # dlt source + pipeline (OpenAlex -> Postgres raw)
├── chunking.py                  # abstract -> chunk(s)
├── embeddings.py                # wrapper FastEmbed
├── indexing.py                  # Postgres raw -> Qdrant
├── retrieval.py                 # búsqueda dense en Qdrant
├── rag.py                       # prompt + llamada a Claude + citas
├── evaluation/                  # ground truth, retrieval eval, RAG eval (LLM-as-judge)
└── monitoring/                  # schema.sql + helpers Postgres para logging
app/streamlit_app.py             # chat UI
grafana/provisioning/            # datasource + dashboard de Postgres
scripts/run_pipeline.py          # orquesta ingest -> index
tests/                           # unit tests (chunking, parsing OpenAlex, filtros de retrieval)
```

## Cómo correrlo

### 1. Instalar dependencias

```bash
make install          # crea .venv e instala el paquete en modo editable
cp .env.example .env  # completar ANTHROPIC_API_KEY y OPENALEX_EMAIL
```

### 2. Levantar servicios de desarrollo (Qdrant + Postgres)

```bash
make dev-up
make init-db   # crea el schema "app" (conversations, messages, feedback, eval_runs)
```

### 3. Ingestar y indexar los papers

```bash
make ingest    # dlt: OpenAlex -> Postgres (schema raw.openalex_works)
make index     # chunking + embeddings + upsert a Qdrant
```

### 4. Evaluación (retrieval + RAG)

```bash
make ground-truth     # Claude genera preguntas por chunk -> data/ground_truth.jsonl
make eval-retrieval   # hit-rate@k y MRR@k
make eval-rag         # LLM-as-judge sobre una muestra de preguntas
```

Los resultados agregados quedan logueados en `app.eval_runs` para verse en Grafana.

### 5. Probar el chat

```bash
make app   # streamlit run app/streamlit_app.py
```

### 6. Stack completo con docker-compose (al final)

Una vez validado todo localmente:

```bash
make up   # docker compose up --build: qdrant + postgres + grafana + app
```

- App: http://localhost:8501
- Grafana: http://localhost:3000 (admin/admin)

### Tests

```bash
make test
```

## Notas

- Los embeddings corren 100% local (FastEmbed, sin API key) para abaratar costos; Claude solo se usa para el chat, la generación de ground truth y el juez de evaluación, con el modelo económico (Haiku) por defecto.
- El polite pool de OpenAlex requiere `OPENALEX_EMAIL` en `.env` (mejora la confiabilidad de la API, no hace falta API key).
