from abc import ABC, abstractmethod

from .models import Document, Chunk


class BaseChunker(ABC):
    def __init__(self, max_chunk_size: int) -> None:
        self._max_chunk_size = max_chunk_size

    @abstractmethod
    def chunk_document(self, doc: Document) -> list[Chunk]:
        pass

    def chunk_documents(self, docs: list[Document]) -> list[Chunk]:
        raise NotImplementedError(
            "chunk_documents is not implemented yet"
            )


class FixedSizeChunker(BaseChunker):
    def __init__(self, max_chunk_size: int) -> None:
        super().__init__(max_chunk_size)

    def chunk_document(self, doc: Document) -> list[Chunk]:
        raise NotImplementedError(
            "chunk_document is not implemented yet"
        )