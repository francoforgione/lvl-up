from functools import lru_cache

from fastembed import TextEmbedding

from lvlup.config import get_settings


@lru_cache(maxsize=1)
def get_embedder() -> TextEmbedding:
    settings = get_settings()
    return TextEmbedding(model_name=settings.embedding_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    embedder = get_embedder()
    return [vector.tolist() for vector in embedder.embed(texts)]
