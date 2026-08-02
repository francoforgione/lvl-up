# Lvl Up Coach

Proyecto final de [LLM Zoomcamp 2026](https://github.com/DataTalksClub/llm-zoomcamp) (DataTalksClub).

Un coach RAG de **bienestar digital y hábitos** que responde preguntas como:

- "¿Cómo afecta el scroll pasivo al foco?"
- "¿Cómo mejoro mi HRV?"
- "¿Sirve la zona 2 para la función ejecutiva?"

...citando evidencia real de papers científicos indexados desde [OpenAlex](https://openalex.org/).

## Problema

La info sobre foco, hábitos, bienestar digital y recuperación de patrones compulsivos está
dispersa entre blogs sin respaldo, hilos de Reddit y videos de YouTube — mezclando consejo real
con anécdota. Cuando alguien busca entender algo puntual ("¿el scroll pasivo daña la atención?",
"¿sirve la zona 2 para la función ejecutiva?", "¿qué dice la ciencia sobre romper un hábito
compulsivo?"), rara vez encuentra una respuesta que cite la evidencia real detrás.

**Lvl Up Coach** resuelve esto con un chat que responde preguntas de bienestar digital, hábitos y
salud cognitiva citando papers científicos reales — no conocimiento general del LLM. Está pensado
para cualquiera que quiera entender y mejorar su relación con la tecnología, su atención o sus
hábitos con base en evidencia, sin tener que leer papers académicos por su cuenta. Es el proyecto
final de LLM Zoomcamp 2026 y a la vez el núcleo conversacional de un producto de coaching cognitivo
más amplio (chat 24/7 + monitoreo de hábitos + señales fisiológicas) que se está desarrollando por
fuera de este curso.

## Capstone checklist

Mapeo directo a los 5 bloques que pide el capstone del curso, cada uno con el código y la sección
de este README donde está resuelto — pensado como índice para navegar el repo sin tener que
adivinar dónde vive cada pieza.

| Requisito del capstone | Qué hacemos | Dónde |
|---|---|---|
| **Searchable knowledge base** — elegir dataset, ingerir, limpiar y guardar para retrieval | OpenAlex → dlt → Postgres (`raw`) → chunking → embeddings dense + sparse → Qdrant | [`ingestion/`](src/lvlup/ingestion/), [`chunking.py`](src/lvlup/chunking.py), [`embeddings.py`](src/lvlup/embeddings.py), [`indexing.py`](src/lvlup/indexing.py) — ver [Cómo correrlo](#cómo-correrlo), paso 3 |
| **Retrieval pipeline** — retrieve context, assemble prompt, call LLM, grounded answers | Bucle agéntico: Claude decide cuándo y qué buscar, arma la respuesta solo con las excerpts recuperadas y cita fuente | [`rag.py`](src/lvlup/rag.py), [`tools.py`](src/lvlup/tools.py), [`retrieval.py`](src/lvlup/retrieval.py) — ver [Agentic RAG](#agentic-rag-module-1-del-curso) |
| **Evaluation process** — search metrics o LLM-as-a-Judge | hit-rate/MRR comparando dense vs hybrid + LLM-as-judge comparando prompt baseline vs guarded | [`evaluation/`](src/lvlup/evaluation/) — ver [Cómo correrlo](#cómo-correrlo), paso 4 |
| **User-facing interface** — UI o API | Chat Streamlit: respuesta + fuentes citadas + costo/tokens/iteraciones por turno + botón 👍/👎 | [`app/streamlit_app.py`](app/streamlit_app.py) — ver [Cómo correrlo](#cómo-correrlo), paso 5 |
| **Monitoring & feedback loops** — trackear queries, feedback y performance en el tiempo | Cada turno queda en Postgres (tokens, costo, guardrail, chunks citados, feedback); Grafana grafica 8 paneles sobre esos datos | [`monitoring/`](src/lvlup/monitoring/), [`grafana/provisioning/`](grafana/provisioning/) — ver [Guardrails](#guardrails) |

## Stack

| Componente | Tecnología |
|---|---|
| Ingesta | [dlt](https://dlthub.com/) |
| Fuente de datos | [OpenAlex API](https://docs.openalex.org/) (abstracts sobre screen time, digital addiction, HRV, habit formation, zone 2 training, executive function, compulsive sexual behavior/problematic pornography use, prefrontal cortex development) |
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
                chunking + FastEmbed (dense + BM25 sparse)
                              v
                Qdrant (lvlup_papers, vectores nombrados)
                              ^
                              | search_papers (tool)
                              |
Streamlit UI --pregunta + historial--> rag.py (bucle agentico) --(Claude)--> respuesta + citas
      |                                                                          |
      +---- feedback (👍/👎) + tokens/costo/guardrail --------> Postgres (schema app)
                                                                                  |
                                                                          Grafana dashboards
```

El retrieval **no** es un paso fijo: `search_papers` es una herramienta que Claude decide invocar
(una vez, varias veces con queries distintas, en paralelo, o ninguna si el contexto previo alcanza).

La evaluación (ground truth generado por Claude, hit-rate/MRR de retrieval, LLM-as-judge de RAG) también corre contra este mismo esquema `app.eval_runs`, para que las métricas se puedan graficar en Grafana.

## Guardrails

Sin freno, el retrieval siempre devuelve el top-k más cercano, por lejos que esté — una pregunta
sobre inversiones igual trae papers, solo que irrelevantes, y el LLM puede terminar respondiendo
desde conocimiento general en vez de admitir que está fuera de su dominio. Dos capas:

1. **System prompt** ([rag.py](src/lvlup/rag.py)): scope explícito a los 8 temas del corpus, nunca
   responder desde conocimiento general, sin diagnósticos ni consejo médico, tono sin juicio en
   preguntas sobre conductas compulsivas y recaídas, derivar a un profesional ante señales de
   crisis, e ignorar instrucciones embebidas en la pregunta del usuario. En la práctica, esta capa
   sola ya rechaza la mayoría de las preguntas fuera de dominio sin siquiera llamar a la tool.
2. **Filtro por score** ([guardrails.py](src/lvlup/guardrails.py)): si el prompt no alcanza y el
   modelo busca igual, `search_papers` descarta los chunks por debajo de `MIN_RELEVANCE_SCORE`
   (default `0.68`). Medido contra el corpus real: preguntas en dominio puntúan 0.75-0.85,
   fuera de dominio 0.54-0.64 — el umbral cae en el hueco entre ambos grupos.

   Este filtro corre siempre en modo `dense` aunque el retrieval final use `hybrid`: el score de
   fusión RRF de `hybrid` refleja posición de ranking, no similitud semántica (una pregunta sobre
   "la capital de Francia" puede puntuar más alto que una en dominio), así que no sirve como señal
   de relevancia calibrada. `search_papers` hace dos búsquedas — `dense` solo para decidir si el
   tema está soportado por el corpus, `hybrid` para el contenido que se termina citando.

Verificado contra `app.eval_runs`: en las 472 preguntas del ground truth (generadas *desde* el
corpus, por lo que todas deberían ser respondibles), el filtro por score nunca disparó un falso
rechazo (`guardrail_false_rejection_rate = 0.0`).

## Estructura del repo

```
src/lvlup/
├── config.py                    # settings + lista de temas de OpenAlex
├── ingestion/                   # dlt source + pipeline (OpenAlex -> Postgres raw)
├── chunking.py                  # abstract -> chunk(s)
├── embeddings.py                # wrappers FastEmbed: dense (bge-small) + sparse (BM25)
├── indexing.py                  # Postgres raw -> Qdrant (vectores nombrados: dense + bm25)
├── retrieval.py                 # búsqueda dense o hybrid (dense + BM25, fusión RRF) en Qdrant
├── guardrails.py                # filtro de relevancia por score (evita alucinar fuera de dominio)
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

Para mover el corpus ya ingerido/indexado entre máquinas propias sin volver a pegarle a la API de
OpenAlex ni recalcular embeddings, se puede generar un dump local (`pg_dump` + snapshot de Qdrant)
y copiarlo a `data/migration/` — no viaja en el repo (son binarios, están en `.gitignore`), así que
esto asume que ya tenés esos archivos en tu propia máquina. Con `make dev-up` ya corriendo:

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
make eval-retrieval   # hit-rate@k y MRR@k, dense vs hybrid
make eval-rag RAG_EVAL_ARGS="--prompt guarded"    # LLM-as-judge
make eval-rag RAG_EVAL_ARGS="--prompt baseline"   # idem, con el prompt sin guardrails
```

Los resultados agregados quedan logueados en `app.eval_runs` para verse en Grafana. Se evaluaron
dos approaches de cada tipo y se dejó el que ganó como default:

**Retrieval — dense vs hybrid** (472 preguntas, top-5):

| Modo | hit-rate | MRR |
|---|---|---|
| dense (solo embeddings) | 0.750 | 0.574 |
| **hybrid (dense + BM25, fusión RRF)** | **0.790** | **0.608** |

`hybrid` gana y es el default en [retrieval.py](src/lvlup/retrieval.py) (`search(..., mode="hybrid")`).
Nota: el gate de relevancia de los guardrails corre igual en modo `dense` — el score de fusión RRF
no es una señal de relevancia calibrada, ver [Guardrails](#guardrails).

**RAG — prompt baseline vs guarded** (30 preguntas, muestra fija):

| Prompt | RELEVANT (todas) | Rechazos sin evidencia | RELEVANT (solo entre las respondidas) |
|---|---|---|---|
| baseline (sin guardrails) | 53.3% | 40.0% | 88.9% |
| guarded (con guardrails) | 36.7% | 70.0% | 88.9% |

Con el corpus ampliado, dos temas nuevos (`prefrontal_cortex_development`, con la query de OpenAlex
más genérica del set) trajeron bastante ruido — papers de neurociencia del desarrollo o incluso de
cambio climático que matchean por overlap de palabras sueltas, no por tema real. `guarded` rechaza
esos casos con más frecuencia que `baseline` porque su instrucción de "nunca responder sin evidencia
del corpus" es más estricta — el juez no distingue "rechazo correcto a un paper fuera de tema" de
"mala respuesta", así que penaliza el ratio bruto de `guarded` sin que sea una regresión real.
La métrica que sí es comparable de forma justa (relevancia *cuando el agente decide responder*)
queda empatada en 88.9% — ninguno de los dos prompts sacrifica calidad cuando responde; la diferencia
es cuán seguido decide no hacerlo, que es una decisión de diseño (menos alucinación) a costa de
cobertura. Se mantiene `guarded` como default por eso: preferimos que se abstenga antes que invente.

### 5. Probar el chat

```bash
make app   # streamlit run app/streamlit_app.py
```

### 6. Stack completo con docker-compose

```bash
make up   # docker compose up --build: qdrant + postgres + grafana + app
```

- App: http://localhost:8501
- Grafana: http://localhost:3000 (admin/admin)

`docker-compose.yml` reutiliza los mismos volúmenes nombrados que `docker-compose.dev.yml` (mismo
nombre de proyecto), así que si ya ingeriste/indexaste con `make dev-up`, el stack completo arranca
con los datos ya cargados — no hace falta repetir `make ingest`/`make index`.

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
- `requirements.txt` (lo que instala el `Dockerfile`) tiene versiones fijas, no rangos, para que el build sea reproducible.
