from qdrant_client.models import (
    FieldCondition,
    Filter,
    FusionQuery,
    MatchValue,
    Prefetch,
)

from lvlup.config import get_settings
from lvlup.embeddings import embed_sparse, embed_texts
from lvlup.indexing import DENSE_VECTOR, SPARSE_VECTOR, get_client


def _build_filter(topic: str | None) -> Filter | None:
    if not topic:
        return None
    return Filter(must=[FieldCondition(key="topic_query", match=MatchValue(value=topic))])


def _point_to_result(point) -> dict:
    payload = point.payload
    return {
        "chunk_id": payload.get("chunk_id"),
        "score": point.score,
        "text": payload.get("text"),
        "title": payload.get("title"),
        "doi": payload.get("doi"),
        "publication_year": payload.get("publication_year"),
        "landing_page_url": payload.get("landing_page_url"),
    }


def search(query: str, top_k: int | None = None, topic: str | None = None, mode: str = "hybrid") -> list[dict]:
    """Search the corpus. `mode` is "dense" (embeddings only) or "hybrid" (+ BM25, RRF-fused)."""
    settings = get_settings()
    top_k = top_k or settings.retrieval_top_k
    client = get_client()
    query_filter = _build_filter(topic)
    dense_vector = embed_texts([query])[0]

    if mode == "dense":
        results = client.query_points(
            collection_name=settings.qdrant_collection,
            query=dense_vector,
            using=DENSE_VECTOR,
            query_filter=query_filter,
            limit=top_k,
        )
    elif mode == "hybrid":
        sparse_vector = embed_sparse([query])[0]
        results = client.query_points(
            collection_name=settings.qdrant_collection,
            prefetch=[
                Prefetch(query=dense_vector, using=DENSE_VECTOR, filter=query_filter, limit=top_k * 2),
                Prefetch(query=sparse_vector, using=SPARSE_VECTOR, filter=query_filter, limit=top_k * 2),
            ],
            query=FusionQuery(fusion="rrf"),
            query_filter=query_filter,
            limit=top_k,
        )
    else:
        raise ValueError(f"Unknown search mode: {mode!r}")

    return [_point_to_result(point) for point in results.points]
