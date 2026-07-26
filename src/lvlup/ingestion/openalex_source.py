from typing import Iterator

import dlt
import requests

OPENALEX_WORKS_URL = "https://api.openalex.org/works"


def reconstruct_abstract(inverted_index: dict | None) -> str:
    """OpenAlex stores abstracts as {word: [positions]} to save space; rebuild the text."""
    if not inverted_index:
        return ""
    positions: list[tuple[int, str]] = []
    for word, idxs in inverted_index.items():
        for idx in idxs:
            positions.append((idx, word))
    positions.sort(key=lambda pair: pair[0])
    return " ".join(word for _, word in positions)


def _fetch_page(query: str, cursor: str, per_page: int, email: str) -> dict:
    params = {
        "filter": f"title_and_abstract.search:{query}",
        "per-page": per_page,
        "cursor": cursor,
        "mailto": email,
    }
    response = requests.get(OPENALEX_WORKS_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


@dlt.resource(name="openalex_works", write_disposition="merge", primary_key="id", max_table_nesting=0)
def openalex_works(topics: dict[str, str], per_topic_limit: int, email: str) -> Iterator[dict]:
    for topic_key, query in topics.items():
        fetched = 0
        cursor = "*"
        while cursor and fetched < per_topic_limit:
            per_page = min(200, per_topic_limit - fetched)
            data = _fetch_page(query, cursor, per_page, email)
            results = data.get("results", [])
            if not results:
                break

            for work in results:
                abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
                if not abstract:
                    continue
                primary_location = work.get("primary_location") or {}
                yield {
                    "id": work["id"],
                    "title": work.get("title"),
                    "abstract": abstract,
                    "doi": work.get("doi"),
                    "publication_year": work.get("publication_year"),
                    "cited_by_count": work.get("cited_by_count"),
                    "authors": [
                        authorship["author"]["display_name"]
                        for authorship in work.get("authorships", [])
                        if authorship.get("author")
                    ],
                    "concepts": [concept["display_name"] for concept in work.get("concepts", [])],
                    "topic_query": topic_key,
                    "landing_page_url": primary_location.get("landing_page_url"),
                }
                fetched += 1

            cursor = data.get("meta", {}).get("next_cursor")
            if len(results) < per_page:
                break


@dlt.source
def openalex_source(topics: dict[str, str], per_topic_limit: int, email: str):
    yield openalex_works(topics, per_topic_limit, email)
