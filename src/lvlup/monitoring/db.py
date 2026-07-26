from contextlib import contextmanager
from pathlib import Path

import psycopg2

from lvlup.config import get_settings


@contextmanager
def get_connection():
    settings = get_settings()
    conn = psycopg2.connect(settings.postgres_dsn)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_schema() -> None:
    schema_sql = (Path(__file__).parent / "schema.sql").read_text()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(schema_sql)


def log_conversation_start(session_id: str) -> str:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO app.conversations (session_id) VALUES (%s) RETURNING id",
                (session_id,),
            )
            return str(cur.fetchone()[0])


def log_message(
    conversation_id: str,
    role: str,
    content: str,
    model: str | None = None,
    latency_ms: int | None = None,
) -> str:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO app.messages (conversation_id, role, content, model, latency_ms)
                   VALUES (%s, %s, %s, %s, %s) RETURNING id""",
                (conversation_id, role, content, model, latency_ms),
            )
            return str(cur.fetchone()[0])


def log_retrieved_chunks(message_id: str, chunks: list[dict]) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            for rank, chunk in enumerate(chunks, start=1):
                cur.execute(
                    """INSERT INTO app.retrieved_chunks (message_id, chunk_id, rank, score)
                       VALUES (%s, %s, %s, %s)""",
                    (message_id, chunk["chunk_id"], rank, chunk["score"]),
                )


def log_feedback(message_id: str, rating: int) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO app.feedback (message_id, rating) VALUES (%s, %s)",
                (message_id, rating),
            )


def log_eval_metric(run_type: str, metric_name: str, metric_value: float) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO app.eval_runs (run_type, metric_name, metric_value)
                   VALUES (%s, %s, %s)""",
                (run_type, metric_name, metric_value),
            )
