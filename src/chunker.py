from abc import ABC, abstractmethod

from tqdm import tqdm

from .models import Document, Chunk


class BaseChunker(ABC):
    def __init__(self, max_chunk_size: int) -> None:
        if (not isinstance(max_chunk_size, int)
                or isinstance(max_chunk_size, bool)):
            raise TypeError(
                f"max_chunk_size must be a valid integer. Got {max_chunk_size}"
            )

        if max_chunk_size <= 0:
            raise ValueError(
                f"max_chunk_size must be at least 1. Got {max_chunk_size}"
            )

        self._max_chunk_size = max_chunk_size

    @abstractmethod
    def chunk_document(self, doc: Document) -> list[Chunk]:
        pass

    def chunk_documents(self, docs: list[Document]) -> list[Chunk]:
        chunk_list: list[Chunk] = []
        for doc in tqdm(docs, desc="Chunking files", unit="file"):
            chunk_list.extend(self.chunk_document(doc))

        return chunk_list


class FixedSizeChunker(BaseChunker):
    def chunk_document(self, doc: Document) -> list[Chunk]:
        raise NotImplementedError(
            "chunk_document is not implemented yet"
        )
