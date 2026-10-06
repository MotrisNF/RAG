"""Pydantic models exchanged between the stages of the RAG pipeline.

The first group follows the subject's Data Models section; the rest are
internal to the project (documents, chunks, hits and the saved index).
"""
import uuid
from typing import Literal

from pydantic import BaseModel, Field, model_validator

# How ``Generator`` loads the model (``answer --dtype``). Defined here, not
# in ``generator``, so the CLI can use it without importing torch.
DtypeMode = Literal["auto", "float32", "bfloat16"]
RetrieverMode = Literal["bm25", "hybrid"]

# Answer when retrieval finds no source: the model is not run, since with no
# context it could only answer from its own knowledge. Here for the same
# reason as ``DtypeMode``: ``answer`` uses it without importing torch.
NO_SOURCES = "No relevant sources were found in the corpus."


class MinimalSource(BaseModel):
    """A character range of a corpus file.

    Attributes:
        file_path: Path relative to the project root, as in the corpus
            (``data/raw/vllm-0.10.1/...``).
        first_character_index: Start of the range (included).
        last_character_index: End of the range (excluded).
    """

    file_path: str
    first_character_index: int
    last_character_index: int


class Chunk(MinimalSource):
    """A piece of a corpus file, with its text.

    Attributes:
        text: The characters of the file inside the range.
    """

    text: str


class UnansweredQuestion(BaseModel):
    """A question without its reference answer.

    Attributes:
        question_id: Unique id; a new UUID when not given.
        question: The question text.
    """

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A question with its reference sources and answer.

    Attributes:
        sources: Locations of the corpus that answer the question.
        answer: The reference answer.
    """

    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """A dataset of questions, answered or not.

    Attributes:
        rag_questions: The questions of the dataset.
    """

    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """The sources retrieved for one question.

    Attributes:
        question_id: Id of the question.
        question: The question text.
        retrieved_sources: The sources found, best first.
    """

    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """The sources retrieved for one question and the generated answer.

    Attributes:
        answer: The answer generated from the retrieved sources.
    """

    answer: str


class StudentSearchResults(BaseModel):
    """Output of ``search_dataset``: the results of every question.

    Attributes:
        search_results: One entry per question.
        k: Number of sources requested per question.
    """

    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Output of ``answer_dataset``: results and answers of every question.

    Attributes:
        search_results: One entry per question, with its answer.
        k: Number of sources requested per question.
    """

    search_results: list[MinimalAnswer]
    k: int


class Document(BaseModel):
    """A whole corpus file read into memory.

    Attributes:
        file_path: Path relative to the project root.
        text: Full content of the file.
    """

    file_path: str
    text: str


class CorpusReadResult(BaseModel):
    """What ``CorpusReader.read`` found in the corpus folder.

    Attributes:
        documents: The files read successfully.
        skipped: Paths of the files that could not be read.
    """

    documents: list[Document]
    skipped: list[str]

    @property
    def found(self) -> int:
        """Number of candidate files, read or skipped."""
        return len(self.documents) + len(self.skipped)


class SearchHit(BaseModel):
    """A chunk returned by a search, with its score.

    Attributes:
        chunk: The matching chunk.
        score: Search score; higher is better (BM25, or the combined
            score of the hybrid search).
        chunk_id: Position of the chunk in the index's chunk list; the
            BM25 and vector indexes share it.
    """

    chunk: Chunk
    score: float
    chunk_id: int


class IndexData(BaseModel):
    """The BM25 index as saved on disk.

    Attributes:
        chunks: The indexed chunks; a chunk id is its position here.
        postings: Inverted index, ``term → {chunk id: occurrences}``.
        lengths: Number of tokens of each chunk, by chunk id.
    """

    chunks: list[Chunk]
    postings: dict[str, dict[int, int]]
    lengths: list[int]

    @model_validator(mode="after")
    def _check_consistent(self) -> "IndexData":
        """Reject an index BM25 cannot search.

        One length per chunk and at least one token in total: otherwise
        the average chunk length would be 0 (division by zero).

        Returns:
            The validated index.

        Raises:
            ValueError: The lengths do not match the chunks, or there
                are no tokens.
        """
        if len(self.lengths) != len(self.chunks):
            raise ValueError(
                f"{len(self.lengths)} lengths for {len(self.chunks)} chunks"
            )
        if sum(self.lengths) <= 0:
            raise ValueError("the index has no tokens")
        return self


class EvaluationInform:
    """Result of an evaluation (read-only).

    Attributes:
        recall: ``cutoff → recall``, e.g. ``{1: 0.61, 5: 0.85}``.
        missing: Dataset questions absent from the results (count as 0).
        unknown: Result questions absent from the dataset (ignored).
        failed: Dataset questions whose correct sources were not all found
            within the first k results (missing ones included), in dataset
            order.
    """

    def __init__(self,
                 recall: dict[int, float],
                 missing: list[str],
                 unknown: list[str],
                 failed: list[str]) -> None:
        """Store the results of ``Evaluator.evaluate``.

        Args:
            recall: Recall per cutoff.
            missing: Ids of the questions without results.
            unknown: Ids of the results that are not in the dataset.
            failed: Ids of the questions not fully found.
        """
        self._recall = recall
        self._missing = missing
        self._unknown = unknown
        self._failed = failed

    @property
    def recall(self) -> dict[int, float]:
        """Recall per cutoff."""
        return self._recall

    @property
    def missing(self) -> list[str]:
        """Ids of the dataset questions absent from the results."""
        return self._missing

    @property
    def unknown(self) -> list[str]:
        """Ids of the results that are not in the dataset."""
        return self._unknown

    @property
    def failed(self) -> list[str]:
        """Ids of the questions not fully found within k results."""
        return self._failed


class WindowChunk(BaseModel):
    """The texts embedded for one chunk by the vector index.

    Attributes:
        windows: Each window of the chunk, with the file path words in
            front.
        windows_count: Number of windows (``len(windows)``).
    """

    windows: list[str]
    windows_count: int
