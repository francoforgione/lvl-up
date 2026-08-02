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
                              | search_papers (tool)
                              |
Streamlit UI --pregunta + historial--> rag.py (bucle agentico) --(Claude)--> respuesta + citas
      |                                                                          |
      +---- feedback (👍/👎) + tokens/costo ------------------> Postgres (schema app)
                                                                                  |
                                                                          Grafana dashboards
```

El retrieval **no** es un paso fijo: `search_papers` es una herramienta que Claude decide invocar
(una vez, varias veces con queries distintas, en paralelo, o ninguna si el contexto previo alcanza).

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
├── tools.py                     # tool search_papers + inferencia de schemas desde docstrings
├── rag.py                       # bucle agéntico (function calling + memoria + citas)
├── costs.py                     # contabilidad de tokens y estimación de costo
├── evaluation/                  # ground truth, retrieval eval, RAG eval (LLM-as-judge)
└── monitoring/                  # schema.sql + helpers Postgres para logging
app/streamlit_app.py             # chat UI
grafana/provisioning/            # datasource + dashboard de Postgres
scripts/run_pipeline.py          # orquesta ingest -> index
tests/                           # unit tests (chunking, OpenAlex, retrieval, tools, bucle agéntico)
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

#### Alternativa: restaurar desde un dump (sin re-ingestar/re-embeder)

Para mover el corpus ya ingerido/indexado a otra máquina sin volver a pegarle a la API de OpenAlex ni recalcular embeddings, hay dumps en `data/migration/` (`lvlup_postgres.dump` y `lvlup_papers_qdrant.snapshot`). Con `make dev-up` ya corriendo:

```bash
# Postgres (raw.openalex_works + schema app)
docker cp data/migration/lvlup_postgres.dump lvl-up-postgres-1:/tmp/lvlup_postgres.dump
docker exec lvl-up-postgres-1 pg_restore -U lvlup -d lvlup --clean --if-exists /tmp/lvlup_postgres.dump

# Qdrant (colección lvlup_papers completa, con vectores)
curl -X POST "http://localhost:6333/collections/lvlup_papers/snapshots/upload?priority=snapshot" \
  -H "Content-Type: multipart/form-data" \
  -F "snapshot=@data/migration/lvlup_papers_qdrant.snapshot"
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

## Agentic RAG (Module 1 del curso)

El chat no es un RAG rígido: `search_papers` está expuesta como *tool* y Claude decide cuándo y
cómo buscar dentro de un bucle acotado. Dónde vive cada pieza del módulo:

| Concepto | Implementación |
|---|---|
| RAG tradicional (búsqueda → prompt → respuesta) | Punto de partida, hoy reemplazado por el bucle en [rag.py](src/lvlup/rag.py) |
| Function calling (definición de tools) | `ToolExecutor.tools` en [tools.py](src/lvlup/tools.py) |
| Procesamiento de tool calls | Ejecución + `tool_result` en el bucle de [rag.py](src/lvlup/rag.py) |
| Memoria / historial | Parámetro `history` de `answer()`; la API es stateless, se reenvía todo el hilo |
| Monitoreo de costos y tokens | [costs.py](src/lvlup/costs.py) + columnas `input_tokens`/`output_tokens`/`cost_usd` en `app.messages` |
| Bucle agéntico | `while` con múltiples iteraciones en `answer()` |
| Auto-inferencia de esquemas | `build_tool_schema()` — introspección de type hints y docstring |
| Condiciones de salida | `stop_reason != "tool_use"`, tope `MAX_ITERATIONS`, tool errors como `is_error` |

**Sobre frameworks:** el bucle está escrito a mano a propósito. El SDK de Anthropic ofrece
`client.beta.messages.tool_runner()` (con el decorador `@beta_tool`, que también infiere schemas),
y existen frameworks como LangChain o el Toy AI Kit del curso — todos resuelven esto en menos
líneas, pero esconden justamente lo que el módulo pide entender: el ciclo de decisión, el manejo
de `tool_result` y las condiciones de corte. Para producción, `tool_runner` sería la opción
razonable; acá el objetivo es que el mecanismo quede visible y testeable.

Los tests del bucle ([tests/test_rag_loop.py](tests/test_rag_loop.py)) usan un cliente falso, así
que corren sin API key, sin costo y sin Qdrant levantado.

## Notas

- Los embeddings corren 100% local (FastEmbed, sin API key) para abaratar costos; Claude solo se usa para el chat, la generación de ground truth y el juez de evaluación, con el modelo económico (Haiku) por defecto.
- El polite pool de OpenAlex requiere `OPENALEX_EMAIL` en `.env` (mejora la confiabilidad de la API, no hace falta API key).
- El corpus está en inglés: Claude traduce la query al invocar la tool, lo que evita una llamada extra al LLM por pregunta.
