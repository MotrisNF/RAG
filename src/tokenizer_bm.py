"""Tokenizer shared by indexing and search."""
import re


class Tokenizer:
    """Split text into lowercase terms for BM25.

    Two passes over the lowercase text:

    1. ``PATTERN``: every run of letters or digits
       (``max_model_len`` → ``max``, ``model``, ``len``).
    2. ``SNAKE_PATTERN``: snake_case names kept whole as extra terms
       (``max_model_len``), so a question that quotes a parameter matches
       it exactly, while the parts still match a question that only
       names one of them.

    The passes never produce the same term (the second one always
    contains an underscore, the first one never), so nothing is counted
    twice. The same tokenizer must be used for the chunks and the
    queries, or their terms would never match: rebuild the index after
    changing it.

    Attributes:
        PATTERN: A term is a run of lowercase letters or digits.
        SNAKE_PATTERN: Two or more runs of letters or digits joined by
            single underscores (``__init__`` or ``a__b`` do not match).
        STOPWORDS: Words to drop from the QUERY terms only (not from the
            index). In a code corpus words like ``how`` or ``what`` are
            rare, so BM25 gives them a high idf and they pull in chunks
            that only share the question wording.
    """

    PATTERN = re.compile(r"[a-z0-9]+")
    SNAKE_PATTERN = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)+")
    STOPWORDS: frozenset[str] = frozenset({
        # Question words and auxiliaries
        "what", "why", "where", "when", "which", "who", "how",
        "does", "do", "did", "is", "are", "was",
        "can", "should", "would", "could",
        # Common English words
        "the", "a", "an", "of", "to", "in", "on", "for", "and", "or",
        "with", "by", "it", "its", "this", "that", "be", "i", "you", "my",
        "there", "from", "as", "at", "use", "used", "using",
        # In every chunk path (idf 0)
        "vllm",
    })

    def tokenize(self, text: str) -> list[str]:
        """Return the terms of ``text``, repetitions included.

        Repetitions matter: they are the term frequency of BM25.

        Args:
            text: Any text.

        Returns:
            The plain terms, then the snake_case names, each group in
            text order. Empty when there are none.
        """
        low_text = text.lower()
        return (self.PATTERN.findall(low_text)
                + self.SNAKE_PATTERN.findall(low_text))

    def tokenize_query(self, text: str) -> list[str]:
        return [
                term
                for term in self.tokenize(text)
                if term not in self.STOPWORDS
                ]
