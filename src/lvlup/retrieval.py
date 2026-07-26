from qdrant_client.models import FieldCondition, Filter, MatchValue

from lvlup.config import get_settings
from lvlup.embeddings import embed_texts
from lvlup.indexing import get_client


def _build_filter(topic: str | None) -> Filter | None:
    if not topic:
        return None
    return Filter(must=[FieldCondition(key="topic_query", match=MatchValue(value=topic))])


def search(query: str, top_k: int | None = None, topic: str | None = None) -> list[dict]:
    settings = get_settings()
    top_k = top_k or settings.retrieval_top_k
    client = get_client()
    vector = embed_texts([query])[0]

    results = client.query_points(
        collection_name=settings.qdrant_collection,
        query=vector,
        query_filter=_build_filter(topic),
        limit=top_k,
        with_payload=True,
    )

    return [
        {
            "chunk_id": point.payload.get("chunk_id"),
            "score": point.score,
            "text": point.payload.get("text"),
            "title": point.payload.get("title"),
            "doi": point.payload.get("doi"),
            "publication_year": point.payload.get("publication_year"),
            "landing_page_url": point.payload.get("landing_page_url"),
        }
        for point in results.points
    ]
