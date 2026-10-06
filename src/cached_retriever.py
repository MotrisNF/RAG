from collections.abc import Callable

from .base_retriever import Retriever
from .models import SearchHit


class CachedRetriever(Retriever):
    def __init__(self, load_inner: Callable[[], Retriever]) -> None:
        self._load_inner = load_inner
        self._inner: Retriever | None = None
        self._entries: dict[str, list[SearchHit]] = {}

    def _retriever(self) -> Retriever:
        if self._inner is None:
            self._inner = self._load_inner()
        return self._inner

    @staticmethod
    def _key(query: str, k: int) -> str:
        return f"{k}:{query}"

    def search(self, query: str, k: int) -> list[SearchHit]:
        key = self._key(query, k)
        if key not in self._entries:
            self._entries[key] = self._retriever().search(query, k)
        return list(self._entries[key])