import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _build_metadata(work: dict) -> dict:
    return {
        "title": work.get("title"),
        "doi": work.get("doi"),
        "publication_year": work.get("publication_year"),
        "authors": work.get("authors") or [],
        "topic_query": work.get("topic_query"),
        "landing_page_url": work.get("landing_page_url"),
        "source_id": work["id"],
    }


def chunk_abstract(work: dict, max_chars: int = 1200) -> list[Chunk]:
    """Most OpenAlex abstracts are short enough to be a single chunk; longer ones
    get split on sentence boundaries so no chunk exceeds max_chars."""
    text = (work.get("abstract") or "").strip()
    if not text:
        return []

    metadata = _build_metadata(work)

    if len(text) <= max_chars:
        return [Chunk(chunk_id=f"{work['id']}::0", doc_id=work["id"], text=text, metadata=metadata)]

    chunks: list[Chunk] = []
    current = ""
    idx = 0
    for sentence in _split_sentences(text):
        if current and len(current) + len(sentence) + 1 > max_chars:
            chunks.append(Chunk(chunk_id=f"{work['id']}::{idx}", doc_id=work["id"], text=current.strip(), metadata=metadata))
            idx += 1
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(Chunk(chunk_id=f"{work['id']}::{idx}", doc_id=work["id"], text=current.strip(), metadata=metadata))
    return chunks
