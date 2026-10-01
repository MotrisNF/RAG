"""BM25 search index over the corpus chunks.

The index is built once by the ``index`` command, saved to disk and loaded
by the search commands, so the corpus is never re-read to answer a query.
"""
from collections import Counter
import heapq
from math import log
from pathlib import Path

from pydantic import ValidationError
from tqdm import tqdm

from .exceptions import (DatasetError, IndexBuildError, IndexNotFoundError,
                         PathError)
from .models import Chunk, IndexData, SearchHit
from .tokenizer_bm import Tokenizer


class BM25Index:
    """Lexical search over chunks with the BM25 ranking function.

    Always created complete: from chunks with ``build`` or from disk with
    ``load``; both end in ``__init__`` with the data already computed.

    Attributes:
        K1: Term-frequency saturation. Higher values let repeated terms
            count for more.
        B: Length normalisation, from 0 (ignore chunk length) to 1 (fully
            relative to the average length).
        FILE_NAME: Name of the index file inside the index folder.
    """

    K1: float = 1.5
    B: float = 0.75
    FILE_NAME: str = "index.json"

    def __init__(self,
                 chunks: list[Chunk],
                 postings: dict[str, dict[int, int]],
                 lengths: list[int]
                 ) -> None:
        """Store an already computed index.

        Use ``build`` or ``load`` instead of calling this directly.

        Args:
            chunks: The indexed chunks; a chunk id is its position here.
            postings: Inverted index, ``term → {chunk id: occurrences}``.
            lengths: Number of tokens of each chunk, by chunk id. Their sum
                must be positive (``build`` guarantees it).
        """
        self._tokenizer = Tokenizer()
        self._chunks = chunks
        self._postings = postings
        self._lengths = lengths
        self._avg_length: float = sum(self._lengths) / len(self._lengths)
        self._n_chunks: int = len(self._chunks)

    @classmethod
    def build(cls, chunks: list[Chunk]) -> "BM25Index":
        """Build the index by tokenizing every chunk once.

        Each chunk's tokens are counted, added to the postings and then
        discarded: only the counts and the lengths are kept.

        Args:
            chunks: The chunks to index, in their final order.

        Returns:
            A complete index, ready to search or save.

        Raises:
            IndexBuildError: There is nothing to index (no chunks, or no
                chunk with a single token).
        """
        tokenizer = Tokenizer()
        posting: dict[str, dict[int, int]] = {}
        lengths: list[int] = []

        progress = tqdm(chunks, desc="Tokenizing chunks", unit="chunk")
        for chunk_id, chunk in enumerate(progress):
            tokens = tokenizer.tokenize(chunk.text)
            # EXPERIMENT: also index the file path, so every chunk of e.g.
            # ".../models/keye.py" gets the term "keye" even when its text
            # never says it. Only the indexed terms change: the chunk text
            # and its offsets stay the same. Common parts of the path
            # ("data", "raw", "vllm"...) appear in every chunk, so their IDF
            # is ~0 and they barely count. Uncomment, rebuild the index
            # (`index`) and compare with `evaluate`.
            # tokens += tokenizer.tokenize(chunk.file_path)
            lengths.append(len(tokens))
            for term, count in Counter(tokens).items():
                posting.setdefault(term, {})[chunk_id] = count

        if sum(lengths) == 0:
            raise IndexBuildError(
                "Nothing to index: the corpus has no searchable text "
                f"({len(chunks)} chunks, 0 tokens)"
            )

        return cls(chunks, posting, lengths)

    def _idf(self, term: str) -> float:
        """Rarity of a term: ``ln((N - df + 0.5) / (df + 0.5) + 1)``.

        ``N`` is the number of chunks and ``df`` the number of chunks that
        contain the term. Rare terms get high values; a term present in
        every chunk gets almost 0, never a negative value.

        Args:
            term: A term that IS in the postings (``search`` checks it).

        Returns:
            The inverse document frequency of the term.
        """
        df: int = len(self._postings[term])
        return log((self._n_chunks - df + 0.5) / (df + 0.5) + 1)

    def _term_score(self, idf: float, tf: int, chunk_id: int) -> float:
        """BM25 contribution of one query term to one chunk.

        ``idf × tf × (K1 + 1) / (tf + K1 × (1 − B + B × length / avg))``

        - ``idf`` rewards rare terms.
        - The ``tf`` fraction grows with each occurrence but saturates:
          it never exceeds ``K1 + 1`` however often the term repeats.
        - ``length / avg`` lowers the score of long chunks, where a term
          is more likely to appear by chance (``B`` sets how much).

        Args:
            idf: Rarity of the term, from ``_idf``.
            tf: Occurrences of the term in the chunk.
            chunk_id: Position of the chunk in ``self._chunks``.

        Returns:
            The amount to add to the chunk's score.
        """
        length_ratio = self._lengths[chunk_id] / self._avg_length
        normaliser = self.K1 * (1 - self.B + self.B * length_ratio)
        return idf * tf * (self.K1 + 1) / (tf + normaliser)

    def search(self, query: str, k: int) -> list[SearchHit]:
        """Return the ``k`` chunks that best match the query.

        Each distinct query term adds its ``_term_score`` to every chunk
        that contains it; chunks with no query term are never scored.
        Ties are broken by chunk id, so results never depend on the order
        in which the query terms are visited.

        Args:
            query: Free text; it goes through the same tokenizer as the
                chunks.
            k: Maximum number of results.

        Returns:
            Up to ``k`` hits, best first. Empty when no term matches.
        """
        tokenized_query = set(self._tokenizer.tokenize(query))

        scores: dict[int, float] = {}
        for term in tokenized_query:
            if term not in self._postings:
                continue
            idf = self._idf(term)

            for chunk_id, tf in self._postings[term].items():
                scores[chunk_id] = (scores.get(chunk_id, 0.0)
                                    + self._term_score(idf, tf, chunk_id))

        ranked = heapq.nsmallest(
            k, scores.items(), key=lambda item: (-item[1], item[0])
        )

        hits: list[SearchHit] = []

        for chunk_id, score in ranked:
            hits.append(SearchHit(chunk=self._chunks[chunk_id], score=score))

        return hits

    def save(self, index_dir: Path) -> None:
        """Write the index as JSON into ``index_dir / FILE_NAME``.

        The folder is created if needed and a previous index is replaced.

        Args:
            index_dir: Folder where the index is saved.

        Raises:
            PathError: The folder or the file cannot be written.
        """
        try:
            index_dir.mkdir(parents=True, exist_ok=True)

            index = IndexData(
                chunks=self._chunks,
                postings=self._postings,
                lengths=self._lengths)
            path = index_dir / self.FILE_NAME
            path.write_text(index.model_dump_json(), encoding="utf-8")
        except OSError as e:
            raise PathError(index_dir, e) from e

    @classmethod
    def load(cls, index_dir: Path) -> "BM25Index":
        """Read an index saved by ``save``, without recomputing anything.

        Args:
            index_dir: Folder given to ``save``.

        Returns:
            The same index that was saved, ready to search.

        Raises:
            IndexNotFoundError: There is no index file (``index`` was
                never run for this folder).
            PathError: The file exists but cannot be read.
            DatasetError: The file is not a valid index.
        """
        path = index_dir / cls.FILE_NAME
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError as e:
            raise IndexNotFoundError(index_dir) from e
        except OSError as e:
            raise PathError(path, e) from e
        try:
            data = IndexData.model_validate_json(text)
        except ValidationError as e:
            raise DatasetError(str(path), e) from e

        return cls(data.chunks, data.postings, data.lengths)
