import sys

from lvlup.chunking import chunk_abstract
from lvlup.config import get_settings
from lvlup.indexing import index_chunks
from lvlup.ingestion.pipeline import fetch_raw_works
from lvlup.ingestion.pipeline import run as run_ingest


def run_index(recreate: bool = False) -> None:
    settings = get_settings()
    works = fetch_raw_works()
    chunks = []
    for work in works:
        chunks.extend(chunk_abstract(work, settings.chunk_max_chars))
    total = index_chunks(chunks, recreate=recreate)
    print(f"Indexed {total} chunks from {len(works)} works into Qdrant.")


def main() -> None:
    args = sys.argv[1:]
    recreate = "--recreate" in args
    steps = [a for a in args if not a.startswith("--")]
    step = steps[0] if steps else "all"

    if step in ("ingest", "all"):
        run_ingest()
    if step in ("index", "all"):
        run_index(recreate=recreate)


if __name__ == "__main__":
    main()
