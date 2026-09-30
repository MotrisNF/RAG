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
    SEPARATORS: tuple[str, ...] = ("\n\n", "\n", " ")

    def _find_cut(self, text: str, start: int, end: int, size: int) -> int:
        for separator in self.SEPARATORS:
            position = text.rfind(separator, start, end)

            if position > start + size // 2:
                return position + len(separator)

        return end

    def chunk_document(self, doc: Document) -> list[Chunk]:
        text = doc.text
        size = self._max_chunk_size
        chunk_list: list[Chunk] = []

        start = 0

        while start < len(text):
            end = min(start + size, len(text))

            if end < len(text):
                end = self._find_cut(text, start, end, size)

            chunk = text[start:end]
            if chunk.strip():
                chunk_list.append(Chunk(
                    file_path=doc.file_path,
                    first_character_index=start,
                    last_character_index=end,
                    text=chunk
                ))

            start = end

        return chunk_list
