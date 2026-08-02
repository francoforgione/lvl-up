import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, SparseVectorParams, VectorParams

from lvlup.chunking import Chunk
from lvlup.config import get_settings
from lvlup.embeddings import embed_sparse, embed_texts

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "bm25"


def get_client() -> QdrantClient:
    settings = get_settings()
    return QdrantClient(url=settings.qdrant_url)


def ensure_collection(client: QdrantClient, recreate: bool = False) -> None:
    settings = get_settings()
    exists = client.collection_exists(settings.qdrant_collection)
    if exists and not recreate:
        return
    if exists:
        client.delete_collection(settings.qdrant_collection)
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config={
            DENSE_VECTOR: VectorParams(size=settings.embedding_dim, distance=Distance.COSINE)
        },
        sparse_vectors_config={SPARSE_VECTOR: SparseVectorParams()},
    )


def _point_id(chunk_id: str) -> str:
    # Qdrant point ids must be an unsigned int or a UUID; derive a stable UUID from our own chunk id.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def index_chunks(chunks: list[Chunk], batch_size: int = 64, recreate: bool = False) -> int:
    settings = get_settings()
    client = get_client()
    ensure_collection(client, recreate=recreate)

    total = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.text for c in batch]
        dense_vectors = embed_texts(texts)
        sparse_vectors = embed_sparse(texts)
        points = [
            PointStruct(
                id=_point_id(chunk.chunk_id),
                vector={DENSE_VECTOR: dense, SPARSE_VECTOR: sparse},
                payload={"chunk_id": chunk.chunk_id, "text": chunk.text, **chunk.metadata},
            )
            for chunk, dense, sparse in zip(batch, dense_vectors, sparse_vectors)
        ]
        client.upsert(collection_name=settings.qdrant_collection, points=points)
        total += len(points)
    return total
