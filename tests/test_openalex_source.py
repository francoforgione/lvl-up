from lvlup.ingestion.openalex_source import reconstruct_abstract


def test_reconstruct_abstract_orders_words():
    inverted = {"Digital": [0], "wellness": [1], "matters": [2]}
    assert reconstruct_abstract(inverted) == "Digital wellness matters"


def test_reconstruct_abstract_handles_repeated_words():
    inverted = {"focus": [0, 3], "and": [1], "attention": [2]}
    assert reconstruct_abstract(inverted) == "focus and attention focus"


def test_reconstruct_abstract_handles_none():
    assert reconstruct_abstract(None) == ""


def test_reconstruct_abstract_handles_empty_dict():
    assert reconstruct_abstract({}) == ""
