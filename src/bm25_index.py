from collections import Counter
import heapq
from math import log

from tqdm import tqdm

from .exceptions import IndexBuildError
from .models import Chunk, SearchHit
from .tokenizer_bm import Tokenizer


class BM25Index:
    K1: float = 1.5
    B: float = 0.75

    def __init__(self,
                 chunks: list[Chunk],
                 postings: dict[str, dict[int, int]],
                 lengths: list[int]
                 ) -> None:
        self._tokenizer = Tokenizer()
        self._chunks = chunks
        self._postings = postings
        self._lengths = lengths
        self._avg_length: float = sum(self._lengths) / len(self._lengths)
        self._n_chunks: int = len(self._chunks)

    @classmethod
    def build(cls, chunks: list[Chunk]) -> "BM25Index":
        tokenizer = Tokenizer()
        posting: dict[str, dict[int, int]] = {}
        lengths: list[int] = []

        progress = tqdm(chunks, desc="Tokenizing chunks", unit="chunk")
        for chunk_id, chunk in enumerate(progress):
            tokens = tokenizer.tokenize(chunk.text)
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
