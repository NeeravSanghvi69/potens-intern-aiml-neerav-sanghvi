"""
Cheap, deterministic first line of defense against hallucination.

If the best-matching chunk's similarity score is below SIMILARITY_THRESHOLD,
we don't even call the LLM -- we short-circuit and tell the user the docs
don't cover it. This is faster, cheaper, and more reliable than hoping the
LLM's instructions alone will stop it from guessing.

The LLM prompt (see llm.py) is the second line of defense, for cases where
retrieval returns marginally-related chunks that don't actually answer
the question.
"""

SIMILARITY_THRESHOLD = 0.45


def passes_guardrail(hits: list) -> bool:
    if not hits:
        return False
    return hits[0]["similarity"] >= SIMILARITY_THRESHOLD
