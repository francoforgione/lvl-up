import anthropic

from lvlup.config import get_settings
from lvlup.retrieval import search

SYSTEM_PROMPT = """You are Lvl Up Coach, a digital wellness and habits coach.
Answer the user's question using ONLY the provided research excerpts.
Always cite sources inline as (Title, Year) and list them at the end with their DOI/URL.
If the excerpts don't contain a good answer, say so honestly instead of making things up.
Respond in the same language the user asked the question in."""

TRANSLATE_SYSTEM_PROMPT = (
    "Translate the user's message to English for a document search query. "
    "Reply with ONLY the translation, no quotes or extra text. "
    "If it's already in English, return it unchanged."
)


def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[{i}] {chunk.get('title')} ({chunk.get('publication_year', 'n/d')})\n"
            f"DOI/URL: {chunk.get('doi') or chunk.get('landing_page_url') or 'n/d'}\n"
            f"{chunk.get('text')}"
        )
    return "\n\n".join(parts)


def translate_to_english(client: anthropic.Anthropic, model: str, text: str) -> str:
    message = client.messages.create(
        model=model,
        max_tokens=200,
        system=TRANSLATE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": text}],
    )
    return "".join(block.text for block in message.content if block.type == "text").strip()


def answer(question: str, top_k: int | None = None, topic: str | None = None, model: str | None = None) -> dict:
    settings = get_settings()
    chat_model = model or settings.anthropic_model_chat
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    # The corpus (OpenAlex abstracts) and embedding model are English-only, so
    # translate the query before retrieval to keep semantic matching accurate
    # for non-English questions (e.g. Spanish), without changing the embedding model.
    search_query = translate_to_english(client, chat_model, question)
    chunks = search(search_query, top_k=top_k, topic=topic)
    context = build_context(chunks)

    message = client.messages.create(
        model=chat_model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Research excerpts:\n\n{context}\n\nQuestion: {question}"}],
    )
    text = "".join(block.text for block in message.content if block.type == "text")
    return {"answer": text, "chunks": chunks}
