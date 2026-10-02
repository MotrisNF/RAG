"""Split corpus documents into chunks small enough to retrieve."""
from abc import ABC, abstractmethod

from tqdm import tqdm

from .models import Document, Chunk


class BaseChunker(ABC):
    """Common base of the chunkers: size check and batch chunking."""

    def __init__(self, max_chunk_size: int) -> None:
        """Store the maximum chunk size.

        Args:
            max_chunk_size: Maximum length of a chunk, in characters.

        Raises:
            TypeError: ``max_chunk_size`` is not an integer.
            ValueError: ``max_chunk_size`` is lower than 1.
        """
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
        """Split one document into chunks.

        Args:
            doc: The document to split.

        Returns:
            Its chunks, in file order.
        """

    def chunk_documents(self, docs: list[Document]) -> list[Chunk]:
        """Split every document, with a progress bar.

        Args:
            docs: The documents to split.

        Returns:
            The chunks of all the documents, in order.
        """
        chunk_list: list[Chunk] = []
        for doc in tqdm(docs, desc="Chunking files", unit="file"):
            chunk_list.extend(self.chunk_document(doc))

        return chunk_list


class FixedSizeChunker(BaseChunker):
    """Cut text into chunks of at most ``max_chunk_size`` characters.

    Each cut is moved back to the last separator found in the second
    half of the chunk, so chunks tend to end at a blank line, a line or
    a word instead of in the middle of one.

    Attributes:
        SEPARATORS: Preferred cut points, best first.
        CUT_BEFORE: Separators that START the next chunk instead of
            ending the current one. None here: a blank line, a newline or
            a space stays at the end of the chunk it closes.
    """

    SEPARATORS: tuple[str, ...] = ("\n\n", "\n", " ")
    CUT_BEFORE: frozenset[str] = frozenset()

    def _find_cut(self, text: str, start: int, end: int, size: int) -> int:
        """Position where the chunk ``text[start:end]`` should end.

        A separator of ``CUT_BEFORE`` (e.g. a Python definition in
        ``PyChunker``) is cut right after its leading newline, so it
        starts the next chunk instead of ending this one. Here
        ``CUT_BEFORE`` is empty: every cut goes after the separator.

        Args:
            text: The whole document.
            start: Start of the chunk.
            end: Furthest allowed end.
            size: Maximum chunk size.

        Returns:
            For the best separator found in the second half of the chunk:
            the position just before the definition (``CUT_BEFORE``) or
            just after the separator (the others). ``end`` if there is
            none.
        """
        for separator in self.SEPARATORS:
            position = text.rfind(separator, start, end)

            if position > start + size // 2:
                if separator not in self.CUT_BEFORE:
                    return position + len(separator)
                else:
                    return position + 1

        return end

    def chunk_document(self, doc: Document) -> list[Chunk]:
        """Split one document into consecutive chunks.

        Chunks that are only whitespace are dropped.

        Args:
            doc: The document to split.

        Returns:
            Its chunks, in file order, with their character ranges.
        """
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


class PyChunker(FixedSizeChunker):
    """Cut Python code preferably right before a definition.

    Chunks then hold whole classes, functions or methods when they fit,
    instead of starting in the middle of one.

    Attributes:
        SEPARATORS: Preferred cut points, best first: top-level
            definitions (decorators first, so they stay with what they
            decorate), then methods (indented four spaces), then the
            generic blank line, newline and space.
        CUT_BEFORE: The definition separators. The cut goes right after
            their leading newline, so ``@``, ``class`` or ``def`` begins
            the next chunk.
    """

    SEPARATORS: tuple[str, ...] = (
        "\n@",
        "\nclass ",
        "\ndef ",
        "\nasync def ",
        "\n    @",
        "\n    def ",
        "\n    async def ",
        "\n\n",
        "\n",
        " ",
    )
    CUT_BEFORE: frozenset[str] = frozenset({
        "\n@",
        "\nclass ",
        "\ndef ",
        "\nasync def ",
        "\n    @",
        "\n    def ",
        "\n    async def ",
    })


class MdChunker(FixedSizeChunker):
    SEPARATORS: tuple[str, ...] = (
        "\n#",
        "\n##",
        "\n### ",
        "\n\n",
        "\n",
        " ",
    )

    CUT_BEFORE: frozenset[str] = frozenset({
        "\n#",
        "\n##",
        "\n### ",
    })
