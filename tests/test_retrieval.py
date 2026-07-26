from lvlup.retrieval import _build_filter


def test_build_filter_none_when_no_topic():
    assert _build_filter(None) is None


def test_build_filter_matches_topic_query_field():
    query_filter = _build_filter("hrv")
    assert query_filter is not None
    assert query_filter.must[0].key == "topic_query"
    assert query_filter.must[0].match.value == "hrv"
