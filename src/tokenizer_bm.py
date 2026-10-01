"""Tokenizer shared by indexing and search."""
import re


class Tokenizer:
    """Split text into lowercase alphanumeric terms.

    The same tokenizer must be used for the chunks and the queries, or
    their terms would never match.

    Attributes:
        PATTERN: A term is a run of lowercase letters or digits.
    """

    PATTERN = re.compile(r"[a-z0-9]+")

    def tokenize(self, text: str) -> list[str]:
        """Return the terms of ``text`` in order, repetitions included.

        Args:
            text: Any text.

        Returns:
            The lowercase terms; empty when there are none.
        """
        return self.PATTERN.findall(text.lower())
