from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

# OpenAlex search queries used to build the paper corpus, keyed by a short topic id
# used later for Qdrant payload filtering.
TOPICS: dict[str, str] = {
    "screen_time_focus": "screen time attention focus",
    "digital_addiction": "smartphone digital addiction",
    "hrv": "heart rate variability",
    "habit_formation": "habit formation behavior change",
    "zone2_training": "zone 2 training moderate intensity exercise",
    "executive_function": "executive function cognitive control exercise",
    "compulsive_sexual_behavior": "compulsive sexual behavior problematic pornography use",
    "prefrontal_cortex_development": "prefrontal cortex development impulse control",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    anthropic_api_key: str = ""
    anthropic_model_chat: str = "claude-haiku-4-5-20251001"
    anthropic_model_eval: str = "claude-haiku-4-5-20251001"

    openalex_email: str = ""
    per_topic_limit: int = 180

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "lvlup"
    postgres_password: str = "lvlup"
    postgres_db: str = "lvlup"

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "lvlup_papers"

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    chunk_max_chars: int = 1200
    retrieval_top_k: int = 5
    # Similarity floor below which retrieved chunks are treated as irrelevant.
    min_relevance_score: float = 0.68

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
