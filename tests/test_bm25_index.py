"""Tests for ``src.tokenizer_bm.Tokenizer`` and ``src.bm25_index.BM25Index``.

Scores are checked against ``Reference``, an independent, deliberately
naive BM25 written straight from the formula, so a mistake in the index
cannot hide behind the same mistake in the test.
"""
import json
import os
import subprocess
import sys
import time
from collections import Counter
from math import log

import pytest

from src.bm25_index import BM25Index
from src.chunker import FixedSizeChunker
from src.exceptions import IndexBuildError, RagError
from src.models import Chunk, SearchHit
from src.reader import CorpusReader
from src.tokenizer_bm import Tokenizer

REAL_RAW = CorpusReader.PROJECT_ROOT / "data" / "raw"
DATASETS = CorpusReader.PROJECT_ROOT / "data" / "datasets"


def make_chunks(*texts: str) -> list[Chunk]:
    """One chunk per text, each in its own fake file."""
    return [Chunk(file_path=f"data/raw/{i}.md", first_character_index=0,
                  last_character_index=len(text), text=text)
            for i, text in enumerate(texts)]


class Reference:
    """Naive BM25 straight from the formula, for comparison."""

    def __init__(self, texts: list[str]) -> None:
        """Tokenize every text once."""
        tokenizer = Tokenizer()
        self.docs = [tokenizer.tokenize(t) for t in texts]
        self.n = len(self.docs)
        self.avg = sum(len(d) for d in self.docs) / self.n

    def idf(self, term: str) -> float:
        """``ln((N - df + 0.5) / (df + 0.5) + 1)``."""
        df = sum(1 for d in self.docs if term in d)
        return log((self.n - df + 0.5) / (df + 0.5) + 1)

    def score(self, query: str, doc_id: int) -> float:
        """Sum over the distinct query terms present in the document."""
        doc = self.docs[doc_id]
        counts = Counter(doc)
        total = 0.0
        for term in set(Tokenizer().tokenize(query)):
            tf = counts[term]
            if tf == 0:
                continue
            norm = BM25Index.K1 * (1 - BM25Index.B
                                   + BM25Index.B * len(doc) / self.avg)
            total += self.idf(term) * tf * (BM25Index.K1 + 1) / (tf + norm)
        return total


class TestTokenizer:
    """Text → list of lowercase words."""

    @pytest.mark.parametrize("text, tokens", [
        ("How do I load a LoRA adapter?",
         ["how", "do", "i", "load", "a", "lora", "adapter"]),
        ("max_model_len", ["max", "model", "len"]),
        ("vllm.entrypoints.openai", ["vllm", "entrypoints", "openai"]),
        ("--max-num-seqs=256", ["max", "num", "seqs", "256"]),
        ("fp8 and int4", ["fp8", "and", "int4"]),
        ("lora LoRA LORA", ["lora", "lora", "lora"]),
        ("", []),
        ("¿? --- ``` !!", []),
        ("   \n\t ", []),
    ])
    def test_tokenize(self, text: str, tokens: list[str]) -> None:
        """Lowercase, split on anything that is not a letter or digit."""
        assert Tokenizer().tokenize(text) == tokens

    def test_keeps_order_and_repeats(self) -> None:
        """Order and repetitions are kept (counts matter for BM25)."""
        assert Tokenizer().tokenize("b a b") == ["b", "a", "b"]

    def test_non_ascii_letters_split(self) -> None:
        """Only ``a-z``: accented letters act as separators."""
        assert Tokenizer().tokenize("café") == ["caf"]

    def test_pattern_compiled_once(self) -> None:
        """The pattern is a shared, precompiled class attribute."""
        assert Tokenizer.PATTERN is Tokenizer().PATTERN


class TestBuild:
    """What ``build`` stores."""

    def test_postings(self) -> None:
        """term → {chunk position: occurrences}."""
        index = BM25Index.build(make_chunks(
            "Load a LoRA adapter. LoRA is fast.", "Load the model."))
        assert index._postings["load"] == {0: 1, 1: 1}
        assert index._postings["lora"] == {0: 2}
        assert index._postings["model"] == {1: 1}
        assert "missing" not in index._postings

    def test_lengths_and_average(self) -> None:
        """Token counts per chunk and their mean."""
        index = BM25Index.build(make_chunks("a b c d", "e f"))
        assert index._lengths == [4, 2]
        assert index._avg_length == 3.0
        assert index._n_chunks == 2

    def test_chunks_kept_in_order(self) -> None:
        """Chunk ids are positions in the given list."""
        chunks = make_chunks("x", "y", "z")
        index = BM25Index.build(chunks)
        assert index._chunks == chunks

    def test_chunk_without_tokens_allowed(self) -> None:
        """A symbols-only chunk is fine if others have text."""
        index = BM25Index.build(make_chunks("---", "real text"))
        assert index._lengths == [0, 2]

    def test_no_chunks(self) -> None:
        """Nothing to index: clear error instead of dividing by zero."""
        with pytest.raises(IndexBuildError, match="Nothing to index"):
            BM25Index.build([])

    def test_only_symbols(self) -> None:
        """Chunks exist but have no tokens: same error."""
        with pytest.raises(IndexBuildError, match="2 chunks, 0 tokens"):
            BM25Index.build(make_chunks("---", "``` ***"))

    def test_error_is_rag_error(self) -> None:
        """``main`` handles it as a project error."""
        with pytest.raises(RagError):
            BM25Index.build([])


class TestIdf:
    """Rarity of a term."""

    TEXTS = ["common rare", "common", "common", "common"]

    def test_rare_beats_common(self) -> None:
        """A term in fewer chunks has a higher idf."""
        index = BM25Index.build(make_chunks(*self.TEXTS))
        assert index._idf("rare") > index._idf("common")

    def test_matches_formula(self) -> None:
        """``ln((N - df + 0.5) / (df + 0.5) + 1)``."""
        index = BM25Index.build(make_chunks(*self.TEXTS))
        assert index._idf("rare") == pytest.approx(log(3.5 / 1.5 + 1))
        assert index._idf("common") == pytest.approx(log(0.5 / 4.5 + 1))

    def test_never_negative(self) -> None:
        """A term in every chunk still has idf ≥ 0."""
        index = BM25Index.build(make_chunks(*self.TEXTS))
        assert index._idf("common") >= 0


class TestTermScore:
    """Contribution of one term to one chunk."""

    @pytest.fixture
    def index(self) -> BM25Index:
        """Chunks of lengths 2, 4 and 12 (average 6)."""
        return BM25Index.build(make_chunks(
            "a b", "a b c d", " ".join(["w"] * 12)))

    def test_matches_formula(self, index: BM25Index) -> None:
        """Same value as the written formula."""
        k1, b = BM25Index.K1, BM25Index.B
        expected = 2.0 * 3 * (k1 + 1) / (3 + k1 * (1 - b + b * 4 / 6))
        assert index._term_score(2.0, 3, 1) == pytest.approx(expected)

    def test_grows_with_tf(self, index: BM25Index) -> None:
        """More occurrences, higher score."""
        scores = [index._term_score(1.0, tf, 1) for tf in (1, 2, 5, 20)]
        assert scores == sorted(scores)

    def test_saturates(self, index: BM25Index) -> None:
        """Never above ``idf × (K1 + 1)`` however many repetitions."""
        assert index._term_score(1.0, 10**6, 1) < BM25Index.K1 + 1

    def test_longer_chunk_scores_less(self, index: BM25Index) -> None:
        """Same tf in a longer chunk counts less."""
        short = index._term_score(1.0, 2, 0)
        long = index._term_score(1.0, 2, 2)
        assert short > long

    def test_proportional_to_idf(self, index: BM25Index) -> None:
        """Doubling idf doubles the contribution."""
        assert index._term_score(2.0, 3, 1) == pytest.approx(
            2 * index._term_score(1.0, 3, 1))


class TestSearch:
    """Ranking of chunks for a query."""

    TEXTS = [
        "Serve the model with the OpenAI compatible server.",
        "Configure the server port and host for the server.",
        "Load a LoRA adapter at runtime.",
        "Unrelated text about tokenizers and chat templates.",
        "The server, the server, the server, the server.",
    ]

    @pytest.fixture
    def index(self) -> BM25Index:
        """Index over ``TEXTS``."""
        return BM25Index.build(make_chunks(*self.TEXTS))

    def test_returns_search_hits(self, index: BM25Index) -> None:
        """Each result is a ``SearchHit`` holding the real chunk."""
        hits = index.search("lora adapter", 3)
        assert all(isinstance(h, SearchHit) for h in hits)
        assert hits[0].chunk.text == self.TEXTS[2]

    def test_scores_match_reference(self, index: BM25Index) -> None:
        """Every score equals the naive reference implementation."""
        reference = Reference(self.TEXTS)
        query = "configure the openai server"
        for hit in index.search(query, 5):
            doc_id = self.TEXTS.index(hit.chunk.text)
            assert hit.score == pytest.approx(reference.score(query, doc_id))

    def test_sorted_by_score(self, index: BM25Index) -> None:
        """Best first."""
        scores = [h.score for h in index.search("the server", 5)]
        assert scores == sorted(scores, reverse=True)

    def test_rare_term_wins(self, index: BM25Index) -> None:
        """"openai" (1 chunk) outweighs repeated "server" (3 chunks)."""
        assert index.search("openai server", 1)[0].chunk.text == \
            self.TEXTS[0]

    def test_saturation_limits_repetition(self, index: BM25Index) -> None:
        """Repeating "server" 4 times does not beat a chunk that also
        matches "configure".
        """
        top = index.search("configure server", 1)[0]
        assert top.chunk.text == self.TEXTS[1]

    def test_only_matching_chunks(self, index: BM25Index) -> None:
        """Chunks sharing no term are not returned, even with a big k."""
        hits = index.search("lora", 10)
        assert [h.chunk.text for h in hits] == [self.TEXTS[2]]

    @pytest.mark.parametrize("k", [1, 2, 3])
    def test_k_limits_results(self, index: BM25Index, k: int) -> None:
        """At most k results."""
        assert len(index.search("the server model", k)) == k

    def test_k_zero(self, index: BM25Index) -> None:
        """k = 0 gives nothing."""
        assert index.search("server", 0) == []

    @pytest.mark.parametrize("query", ["", "   ", "¿?", "---",
                                       "zzzqqq xyzzy"])
    def test_no_results(self, index: BM25Index, query: str) -> None:
        """No tokens or no known term: empty list, no error."""
        assert index.search(query, 5) == []

    def test_case_insensitive(self, index: BM25Index) -> None:
        """The query goes through the same tokenizer."""
        assert index.search("LoRA ADAPTER", 3) == index.search(
            "lora adapter", 3)

    def test_repeated_query_terms_count_once(self, index: BM25Index) -> None:
        """"lora lora lora" scores like "lora"."""
        assert index.search("lora lora lora", 3) == index.search("lora", 3)

    def test_ties_broken_by_position(self) -> None:
        """Identical chunks: the earlier one first, every time."""
        index = BM25Index.build(make_chunks("alpha beta", "gamma",
                                            "alpha beta", "alpha beta"))
        for _ in range(5):
            hits = index.search("beta alpha", 3)
            assert [h.chunk.file_path for h in hits] == [
                "data/raw/0.md", "data/raw/2.md", "data/raw/3.md"]

    def test_ties_stable_across_runs(self) -> None:
        """Tied chunks come in the same order in every Python process.

        The query terms go through a ``set``, whose order changes between
        runs (hash randomization, ``PYTHONHASHSEED``). Chunk 0 only has
        "beta" and chunk 1 only "alpha", with identical scores; without
        the tie-break by position, some runs would return ``[1, 0]``.
        """
        script = (
            "from src.bm25_index import BM25Index\n"
            "from src.models import Chunk\n"
            "chunks = [Chunk(file_path=f'{i}', first_character_index=0,"
            " last_character_index=1, text=t)"
            " for i, t in enumerate(['beta x', 'alpha x'])]\n"
            "index = BM25Index.build(chunks)\n"
            "print([h.chunk.file_path for h in index.search('alpha beta', 2)])"
        )
        orders = set()
        for seed in range(12):
            env = {**os.environ, "PYTHONHASHSEED": str(seed)}
            result = subprocess.run(
                [sys.executable, "-c", script], env=env, check=True,
                capture_output=True, text=True,
                cwd=CorpusReader.PROJECT_ROOT)
            orders.add(result.stdout.strip())
        assert orders == {"['0', '1']"}

    def test_deterministic(self, index: BM25Index) -> None:
        """Same query, same answer."""
        first = index.search("the server model openai", 5)
        assert all(index.search("the server model openai", 5) == first
                   for _ in range(5))


@pytest.fixture(scope="module")
def real_index() -> BM25Index:
    """Index over the whole vLLM corpus, built once."""
    docs = CorpusReader(REAL_RAW).read().documents
    return BM25Index.build(FixedSizeChunker(2000).chunk_documents(docs))


@pytest.mark.skipif(not REAL_RAW.is_dir(), reason="vLLM corpus not present")
class TestRealCorpus:
    """Search over the real corpus."""

    def test_known_question(self, real_index: BM25Index) -> None:
        """A dataset question: the top hit overlaps its ground truth
        (``lora.md`` 4695-6098) with IoU ≥ 0.05, as the moulinette asks.
        """
        hit = real_index.search("What HTTP endpoint is used to dynamically "
                                "load a LoRA adapter in vLLM?", 1)[0]
        assert hit.chunk.file_path.endswith("docs/features/lora.md")
        start = hit.chunk.first_character_index
        end = hit.chunk.last_character_index
        overlap = max(0, min(end, 6098) - max(start, 4695))
        union = max(end, 6098) - min(start, 4695)
        assert overlap / union >= 0.05

    @pytest.mark.skipif(not DATASETS.is_dir(), reason="datasets not present")
    def test_speed_all_public_questions(self,
                                        real_index: BM25Index) -> None:
        """Every public question searched well under the subject's limit
        (90 s for 200 questions).
        """
        questions = []
        for path in (DATASETS / "UnansweredQuestions").glob("*.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            questions += [q["question"] for q in data["rag_questions"]]
        start = time.perf_counter()
        for question in questions:
            assert len(real_index.search(question, 10)) == 10
        assert time.perf_counter() - start < 30
