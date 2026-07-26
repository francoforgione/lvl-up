from lvlup.chunking import chunk_abstract


def _work(abstract: str, work_id: str = "W1") -> dict:
    return {
        "id": work_id,
        "abstract": abstract,
        "title": "Test title",
        "doi": None,
        "publication_year": 2023,
        "authors": [],
        "topic_query": "x",
        "landing_page_url": None,
    }


def test_short_abstract_single_chunk():
    chunks = chunk_abstract(_work("Short abstract about focus."), max_chars=1200)
    assert len(chunks) == 1
    assert chunks[0].doc_id == "W1"
    assert chunks[0].text == "Short abstract about focus."


def test_long_abstract_splits_into_multiple_chunks():
    sentence = "This is a sentence about digital wellness and focus. "
    chunks = chunk_abstract(_work(sentence * 50, work_id="W2"), max_chars=200)
    assert len(chunks) > 1
    assert all(c.doc_id == "W2" for c in chunks)
    assert all(len(c.text) <= 200 for c in chunks)


def test_empty_abstract_returns_no_chunks():
    assert chunk_abstract(_work("")) == []
