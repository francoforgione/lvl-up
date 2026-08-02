"""Relevance gating for retrieved chunks.

The vector store always returns its top-k nearest neighbours, however far away
they are — asking about pizza still yields papers, just bad ones. Dropping
everything below a similarity floor keeps weak matches out of the model's
context entirely, so it can't cite them as if they were evidence.

Measured on the indexed corpus: in-domain questions score 0.75-0.77, off-domain
ones 0.54-0.64. The default floor sits in that gap, leaning towards answering.
"""

OUT_OF_SCOPE_MESSAGE = (
    "No papers in the corpus are relevant to this query. The corpus only covers "
    "screen time and attention, digital addiction, heart rate variability, habit "
    "formation, zone 2 training, and executive function. Tell the user this "
    "question is outside what you can answer from evidence — do not answer it "
    "from general knowledge."
)


def filter_by_relevance(chunks: list[dict], threshold: float) -> list[dict]:
    """Keep only the chunks scoring at or above `threshold`."""
    return [chunk for chunk in chunks if (chunk.get("score") or 0.0) >= threshold]
