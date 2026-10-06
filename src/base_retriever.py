"""The common interface of every retriever: ``search`` and ``flush``."""
from abc import ABC, abstractmethod
from .models import SearchHit


class Retriever(ABC):
    """Anything that returns the best chunks for a question."""

    @abstractmethod
    def search(self, query: str, k: int) -> list[SearchHit]:
        """Return up to ``k`` hits, best first."""

    def flush(self) -> None:
        """Write any pending state to disk.

        Nothing to do by default; ``CachedRetriever`` saves its query
        cache. Callers can always call it without knowing which retriever
        they have.
        """
