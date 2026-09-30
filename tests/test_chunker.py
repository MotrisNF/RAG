"""Tests for ``src.chunker``."""
import random

import pytest

from src.chunker import BaseChunker, FixedSizeChunker
from src.models import Chunk, Document, MinimalSource
from src.reader import CorpusReader

REAL_RAW = CorpusReader.PROJECT_ROOT / "data" / "raw"


class Invariants:
    """Properties every chunking of a document must satisfy."""

    @staticmethod
    def check(doc: Document, chunks: list[Chunk], size: int) -> None:
        """Assert all invariants at once.

        - every chunk belongs to the document and fits in ``size``;
        - its offsets slice exactly its text from the document;
        - chunks are in order, never overlap, never blank;
        - whatever is not covered by a chunk is only whitespace.
        """
        position = 0
        for chunk in chunks:
            start = chunk.first_character_index
            end = chunk.last_character_index
            assert chunk.file_path == doc.file_path
            assert 0 < end - start <= size
            assert doc.text[start:end] == chunk.text
            assert chunk.text.strip()
            assert start >= position
            assert doc.text[position:start].strip() == ""
            position = end
        assert doc.text[position:].strip() == ""


def make_doc(text: str, path: str = "data/raw/a.md") -> Document:
    """Build a document without touching the disk."""
    return Document(file_path=path, text=text)


class TestBaseChunker:
    """Validation and behaviour shared by every chunker."""

    def test_is_abstract(self) -> None:
        """``BaseChunker`` cannot be created: ``chunk_document`` is
        abstract.
        """
        with pytest.raises(TypeError):
            BaseChunker(10)  # type: ignore[abstract]

    @pytest.mark.parametrize("size", ["abc", "5", 2.5, 2.0, None, True,
                                      False])
    def test_size_wrong_type(self, size: object) -> None:
        """Anything but a real ``int`` is a ``TypeError``."""
        with pytest.raises(TypeError, match="valid integer"):
            FixedSizeChunker(size)  # type: ignore[arg-type]

    @pytest.mark.parametrize("size", [0, -1, -2000])
    def test_size_too_small(self, size: int) -> None:
        """Zero or negative is a ``ValueError`` (it would loop forever)."""
        with pytest.raises(ValueError, match="at least 1"):
            FixedSizeChunker(size)

    @pytest.mark.parametrize("size", [1, 2, 2000, 10**9])
    def test_size_valid(self, size: int) -> None:
        """Positive integers are accepted."""
        FixedSizeChunker(size)

    def test_chunk_documents_joins_in_order(self) -> None:
        """All documents are chunked and results keep document order."""
        docs = [make_doc("a " * 30, "data/raw/1.md"),
                make_doc("b " * 30, "data/raw/2.md"),
                make_doc("", "data/raw/3.md")]
        chunker = FixedSizeChunker(20)
        chunks = chunker.chunk_documents(docs)
        expected = [c for d in docs for c in chunker.chunk_document(d)]
        assert chunks == expected
        paths = [c.file_path for c in chunks]
        assert paths == sorted(paths)
        assert "data/raw/3.md" not in paths

    def test_chunk_documents_empty_list(self) -> None:
        """No documents, no chunks."""
        assert FixedSizeChunker(10).chunk_documents([]) == []

    def test_chunks_never_mix_files(self) -> None:
        """Each chunk comes from exactly one document."""
        docs = [make_doc("x" * 25, f"data/raw/{i}.py") for i in range(3)]
        for chunk in FixedSizeChunker(10).chunk_documents(docs):
            path = chunk.file_path
            text = next(d.text for d in docs if d.file_path == path)
            start = chunk.first_character_index
            end = chunk.last_character_index
            assert text[start:end] == chunk.text


class TestSimpleCases:
    """Small documents with an obvious answer."""

    def test_empty(self) -> None:
        """An empty document has no chunks."""
        assert FixedSizeChunker(10).chunk_document(make_doc("")) == []

    @pytest.mark.parametrize("text", [" ", "\n", "\n\n\n", "  \t \n "])
    def test_blank(self, text: str) -> None:
        """Only whitespace: no chunks."""
        assert FixedSizeChunker(10).chunk_document(make_doc(text)) == []

    def test_shorter_than_size(self) -> None:
        """A small document is one chunk with the whole text."""
        doc = make_doc("hello world\n")
        [chunk] = FixedSizeChunker(100).chunk_document(doc)
        assert (chunk.first_character_index,
                chunk.last_character_index) == (0, len(doc.text))
        assert chunk.text == doc.text

    def test_exactly_size(self) -> None:
        """A document of exactly ``size`` characters is one chunk."""
        doc = make_doc("abcdefghij")
        chunks = FixedSizeChunker(10).chunk_document(doc)
        assert [c.text for c in chunks] == ["abcdefghij"]

    def test_offsets_are_exclusive(self) -> None:
        """``last_character_index`` is exclusive, like Python slicing."""
        doc = make_doc("abcdefghij")
        chunks = FixedSizeChunker(4).chunk_document(doc)
        assert [(c.first_character_index, c.last_character_index)
                for c in chunks] == [(0, 4), (4, 8), (8, 10)]

    def test_chunk_is_a_source(self) -> None:
        """A chunk can be used wherever a ``MinimalSource`` is expected."""
        [chunk] = FixedSizeChunker(10).chunk_document(make_doc("abc"))
        assert isinstance(chunk, MinimalSource)

    def test_text_not_modified(self) -> None:
        """Leading spaces and newlines are kept: offsets stay exact."""
        doc = make_doc("   indented\n\n  code\n")
        [chunk] = FixedSizeChunker(100).chunk_document(doc)
        assert chunk.text == doc.text


class TestSeparators:
    """Where the cut is made: paragraph, line, space or hard cut."""

    def test_prefers_paragraph(self) -> None:
        """A paragraph break past half the window wins over lines."""
        text = "a" * 12 + "\n\n" + "b" * 3 + "\n" + "c" * 20
        chunks = FixedSizeChunker(20).chunk_document(make_doc(text))
        assert chunks[0].text == "a" * 12 + "\n\n"

    def test_falls_back_to_line(self) -> None:
        """Without a paragraph break, cut after the last newline."""
        text = "a" * 8 + "\n" + "b" * 6 + "\n" + "c" * 20
        chunks = FixedSizeChunker(20).chunk_document(make_doc(text))
        assert chunks[0].text == "a" * 8 + "\n" + "b" * 6 + "\n"

    def test_falls_back_to_space(self) -> None:
        """Without newlines, cut after the last space."""
        text = "word " * 10
        chunks = FixedSizeChunker(12).chunk_document(make_doc(text))
        assert chunks[0].text == "word word "
        assert all(c.text.endswith(" ") for c in chunks)

    def test_hard_cut(self) -> None:
        """No separator at all: cut exactly at ``size``."""
        text = "x" * 25
        chunks = FixedSizeChunker(10).chunk_document(make_doc(text))
        assert [len(c.text) for c in chunks] == [10, 10, 5]

    def test_separator_stays_in_chunk(self) -> None:
        """The whole separator ends the chunk; nothing is lost."""
        text = "a" * 12 + "\n\n" + "b" * 20
        chunks = FixedSizeChunker(20).chunk_document(make_doc(text))
        assert chunks[0].text.endswith("\n\n")
        assert chunks[1].first_character_index == 14

    def test_early_paragraph_ignored(self) -> None:
        """A paragraph break in the first half is skipped for a later
        newline, so chunks are not tiny.
        """
        text = "a" * 3 + "\n\n" + "b" * 10 + "\n" + "c" * 20
        chunks = FixedSizeChunker(20).chunk_document(make_doc(text))
        assert chunks[0].text == "a" * 3 + "\n\n" + "b" * 10 + "\n"

    def test_early_newline_gives_hard_cut(self) -> None:
        """Only separators in the first half: hard cut at ``size``."""
        text = "ab\n" + "x" * 30
        chunks = FixedSizeChunker(10).chunk_document(make_doc(text))
        assert len(chunks[0].text) == 10

    def test_last_chunk_not_cut(self) -> None:
        """The final piece is not split at a separator, even one past
        half of the window.
        """
        text = "a" * 15 + "\nend"
        chunks = FixedSizeChunker(20).chunk_document(make_doc(text))
        assert [c.text for c in chunks] == [text]


class TestExtremeSizes:
    """Sizes that could break the loop."""

    @pytest.mark.parametrize("text", ["abc", "a\nb\n\nc d", "\n\na b",
                                      "x" * 50])
    def test_size_one(self, text: str) -> None:
        """Size 1 terminates and every chunk has one character."""
        doc = make_doc(text)
        chunks = FixedSizeChunker(1).chunk_document(doc)
        Invariants.check(doc, chunks, 1)
        assert all(len(c.text) == 1 for c in chunks)

    def test_size_two_with_separators(self) -> None:
        """Size 2 with separators everywhere still terminates."""
        doc = make_doc("\n \n\n \n a\n\nb ")
        Invariants.check(doc, FixedSizeChunker(2).chunk_document(doc), 2)

    def test_huge_size(self) -> None:
        """A size far bigger than the text gives one chunk."""
        doc = make_doc("some text\n" * 5)
        assert len(FixedSizeChunker(10**9).chunk_document(doc)) == 1


class TestRandomTexts:
    """Invariants on many generated texts (fixed seeds, reproducible)."""

    PIECES = ["word", "x", "longwordwithoutspaces", " ", "  ", "\n", "\n\n",
              "\t", "é", "日本", "🚀", "def f():", "# Title"]

    @pytest.mark.parametrize("seed", range(40))
    @pytest.mark.parametrize("size", [1, 3, 7, 16, 50, 200])
    def test_invariants(self, seed: int, size: int) -> None:
        """Every invariant holds for random mixes of words and spaces."""
        rng = random.Random(seed)
        text = "".join(rng.choice(self.PIECES)
                       for _ in range(rng.randint(0, 150)))
        doc = make_doc(text)
        Invariants.check(doc, FixedSizeChunker(size).chunk_document(doc),
                         size)


@pytest.fixture(scope="module")
def docs() -> list[Document]:
    """Read the real corpus once for the whole module."""
    return CorpusReader(REAL_RAW).read().documents


@pytest.mark.skipif(not REAL_RAW.is_dir(), reason="vLLM corpus not present")
class TestRealCorpus:
    """The whole vLLM corpus, chunked like ``index`` does."""

    @pytest.mark.parametrize("size", [2000, 500])
    def test_invariants(self, docs: list[Document], size: int) -> None:
        """Every chunk of every file satisfies the invariants."""
        chunker = FixedSizeChunker(size)
        for doc in docs:
            Invariants.check(doc, chunker.chunk_document(doc), size)

    def test_max_length_for_moulinette(self, docs: list[Document]) -> None:
        """No chunk exceeds the moulinette's 2000 characters."""
        chunks = FixedSizeChunker(2000).chunk_documents(docs)
        assert max(c.last_character_index - c.first_character_index
                   for c in chunks) <= 2000
        assert len(chunks) > len(docs)
