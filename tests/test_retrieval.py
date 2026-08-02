import pytest

from lvlup import retrieval
from lvlup.retrieval import _build_filter, search


def test_build_filter_none_when_no_topic():
    assert _build_filter(None) is None


def test_build_filter_matches_topic_query_field():
    query_filter = _build_filter("hrv")
    assert query_filter is not None
    assert query_filter.must[0].key == "topic_query"
    assert query_filter.must[0].match.value == "hrv"


class FakePoint:
    def __init__(self, score: float, chunk_id: str = "c1"):
        self.score = score
        self.payload = {"chunk_id": chunk_id, "text": "t", "title": "T"}


class FakeQdrantClient:
    def __init__(self):
        self.calls: list[dict] = []

    def query_points(self, **kwargs):
        self.calls.append(kwargs)
        return type("Resp", (), {"points": [FakePoint(0.9)]})()


@pytest.fixture(autouse=True)
def stub_embeddings(monkeypatch):
    monkeypatch.setattr(retrieval, "embed_texts", lambda texts: [[0.1, 0.2]])
    monkeypatch.setattr(retrieval, "embed_sparse", lambda texts: ["sparse-vec"])


def test_dense_mode_queries_the_dense_named_vector(monkeypatch):
    client = FakeQdrantClient()
    monkeypatch.setattr(retrieval, "get_client", lambda: client)

    search("hrv", mode="dense")

    assert client.calls[0]["using"] == "dense"
    assert "prefetch" not in client.calls[0]


def test_hybrid_mode_prefetches_both_vectors_and_fuses(monkeypatch):
    client = FakeQdrantClient()
    monkeypatch.setattr(retrieval, "get_client", lambda: client)

    search("hrv", mode="hybrid")

    prefetch = client.calls[0]["prefetch"]
    assert {p.using for p in prefetch} == {"dense", "bm25"}


def test_unknown_mode_raises(monkeypatch):
    monkeypatch.setattr(retrieval, "get_client", lambda: FakeQdrantClient())
    with pytest.raises(ValueError):
        search("hrv", mode="bogus")
