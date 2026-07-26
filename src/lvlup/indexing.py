import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from lvlup.chunking import Chunk
from lvlup.config import get_settings
from lvlup.embeddings import embed_texts


def get_client() -> QdrantClient:
    settings = get_settings()
    return QdrantClient(url=settings.qdrant_url)


def ensure_collection(client: QdrantClient) -> None:
    settings = get_settings()
    if client.collection_exists(settings.qdrant_collection):
        return
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=VectorParams(size=settings.embedding_dim, distance=Distance.COSINE),
    )


def _point_id(chunk_id: str) -> str:
    # Qdrant point ids must be an unsigned int or a UUID; derive a stable UUID from our own chunk id.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def index_chunks(chunks: list[Chunk], batch_size: int = 64) -> int:
    settings = get_settings()
    client = get_client()
    ensure_collection(client)

    total = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vectors = embed_texts([c.text for c in batch])
        points = [
            PointStruct(
                id=_point_id(chunk.chunk_id),
                vector=vector,
                payload={"chunk_id": chunk.chunk_id, "text": chunk.text, **chunk.metadata},
            )
            for chunk, vector in zip(batch, vectors)
        ]
        client.upsert(collection_name=settings.qdrant_collection, points=points)
        total += len(points)
    return total
