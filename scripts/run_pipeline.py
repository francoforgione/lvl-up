import sys

from lvlup.chunking import chunk_abstract
from lvlup.config import get_settings
from lvlup.indexing import index_chunks
from lvlup.ingestion.pipeline import fetch_raw_works
from lvlup.ingestion.pipeline import run as run_ingest


def run_index() -> None:
    settings = get_settings()
    works = fetch_raw_works()
    chunks = []
    for work in works:
        chunks.extend(chunk_abstract(work, settings.chunk_max_chars))
    total = index_chunks(chunks)
    print(f"Indexed {total} chunks from {len(works)} works into Qdrant.")


def main() -> None:
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step in ("ingest", "all"):
        run_ingest()
    if step in ("index", "all"):
        run_index()


if __name__ == "__main__":
    main()
