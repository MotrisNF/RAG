"""Tokenizer shared by indexing and search."""
import re


class Tokenizer:
    """Split text into lowercase terms for BM25.

    Three passes over the lowercase text:

    1. ``PATTERN``: every run of letters or digits
       (``llm.generate()`` → ``llm``, ``generate``).
    2. ``DOT_PATTERN``: dotted names kept whole as extra terms
       (``llm.generate``, ``vllm.entrypoints.openai``), so a question
       that quotes a module or a method matches it exactly, while the
       parts still match a question that only names one of them.
    3. ``SNAKE_PATTERN``: snake_case names kept whole as extra terms
       (``max_model_len``), for the same reason.

    The passes never produce the same term (the second one always
    contains a dot, the third one an underscore and no dot), so nothing
    is counted twice. The same tokenizer must be used for the chunks and
    the queries, or their terms would never match: rebuild the index
    after changing it.

    Attributes:
        PATTERN: A term is a run of lowercase letters or digits.
        DOT_PATTERN: Two or more runs of letters, digits or ``_`` joined
            by dots (also catches versions like ``0.10.1`` and file names
            like ``lora.md``).
        SNAKE_PATTERN: Two or more runs of letters or digits joined by
            single underscores (``__init__`` or ``a__b`` do not match).
    """

    PATTERN = re.compile(r"[a-z0-9]+")
    DOT_PATTERN = re.compile(r"[a-z0-9_]+(?:\.[a-z0-9_]+)+")
    SNAKE_PATTERN = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)+")

    def tokenize(self, text: str) -> list[str]:
        """Return the terms of ``text``, repetitions included.

        Repetitions matter: they are the term frequency of BM25.

        Args:
            text: Any text.

        Returns:
            The plain terms, then the dotted names, then the snake_case
            names, each group in text order. Empty when there are none.
        """
        low_text = text.lower()
        return (self.PATTERN.findall(low_text)
                + self.DOT_PATTERN.findall(low_text)
                + self.SNAKE_PATTERN.findall(low_text))
