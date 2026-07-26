import dlt
import psycopg2
import psycopg2.extras
from dlt.destinations import postgres

from lvlup.config import TOPICS, get_settings
from lvlup.ingestion.openalex_source import openalex_source

RAW_WORKS_COLUMNS = (
    "id, title, abstract, doi, publication_year, cited_by_count, "
    "authors, concepts, topic_query, landing_page_url"
)


def run() -> None:
    settings = get_settings()
    destination = postgres(credentials=settings.postgres_dsn)
    pipeline = dlt.pipeline(pipeline_name="lvlup_openalex", destination=destination, dataset_name="raw")
    source = openalex_source(TOPICS, settings.per_topic_limit, settings.openalex_email)
    load_info = pipeline.run(source)
    print(load_info)


def fetch_raw_works() -> list[dict]:
    """Read back the ingested works from Postgres for chunking/indexing/evaluation."""
    settings = get_settings()
    with psycopg2.connect(settings.postgres_dsn) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"SELECT {RAW_WORKS_COLUMNS} FROM raw.openalex_works")
            return [dict(row) for row in cur.fetchall()]


if __name__ == "__main__":
    run()
