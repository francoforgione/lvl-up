from functools import lru_cache

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client.models import SparseVector

from lvlup.config import get_settings

# Hybrid search's keyword half. Not config: this is the one sparse model that
# pairs with Qdrant's built-in BM25-style scoring, never swapped.
SPARSE_MODEL = "Qdrant/bm25"


@lru_cache(maxsize=1)
def get_embedder() -> TextEmbedding:
    settings = get_settings()
    return TextEmbedding(model_name=settings.embedding_model)


@lru_cache(maxsize=1)
def get_sparse_embedder() -> SparseTextEmbedding:
    return SparseTextEmbedding(model_name=SPARSE_MODEL)


def embed_texts(texts: list[str]) -> list[list[float]]:
    embedder = get_embedder()
    return [vector.tolist() for vector in embedder.embed(texts)]


def embed_sparse(texts: list[str]) -> list[SparseVector]:
    embedder = get_sparse_embedder()
    return [
        SparseVector(indices=v.indices.tolist(), values=v.values.tolist())
        for v in embedder.embed(texts)
    ]
