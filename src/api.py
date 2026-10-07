"""Local HTTP API (bonus 5): the search and answer commands over HTTP.

``RagApi`` keeps one retriever and, after the first answer, one model
loaded between requests, so every request after the first is fast. The
responses are the same pydantic models the CLI prints.
"""
import sys
import threading
from typing import TYPE_CHECKING, Annotated

from fastapi import FastAPI, HTTPException, Query


from .base_retriever import Retriever
from .commands import RagCLI
from .exceptions import RagError
from .models import (NO_SOURCES, DtypeMode, MinimalAnswer, MinimalSource,
                     StudentSearchResults, StudentSearchResultsAndAnswer,
                     UnansweredQuestion)

if TYPE_CHECKING:
    from .generator import Generator


class RagApi:
    """The FastAPI application and the state shared by its requests.

    FastAPI runs each request of a plain ``def`` endpoint in a thread of
    its own, so the shared objects are guarded by two locks (mutexes):
    one for the retriever and its query cache, one for the model. They
    are never held together, so they cannot block each other.

    Attributes:
        DESCRIPTION: Shown at the top of the ``/docs`` page.
        app: The FastAPI application, to run with uvicorn.
    """

    DESCRIPTION = (
        "Question answering over the vLLM codebase. `/search` returns the "
        "best source locations for a question; `/answer` also generates "
        "an answer from them with Qwen/Qwen3-0.6B. The first `/answer` "
        "loads the model (tens of seconds); the next ones reuse it."
    )

    def __init__(self, retriever: Retriever,
                 dtype: DtypeMode = "auto") -> None:
        """Create the application; the model is not loaded yet.

        Args:
            retriever: A loaded retriever (see ``RagCLI._load_retriever``).
            dtype: How the model is loaded by the first ``/answer``:
                ``auto`` chooses from the available memory, ``float32``
                or ``bfloat16`` force it. Set once for the server: the
                model is loaded only once.
        """
        self._retriever = retriever
        self._dtype = dtype
        # Loaded on the first /answer, then kept for the next requests.
        self._generator: "Generator | None" = None
        # FastAPI runs each request in its own thread: one lock so the
        # model is loaded only once and answers one question at a time.
        self._generator_lock = threading.Lock()
        # The retriever (and its cache) is shared by every request: one
        # search, with its flush, at a time. Separate from the generator
        # lock, so /search does not wait for an answer being generated.
        self._retriever_lock = threading.Lock()
        self.app = FastAPI(title="RAG against the machine",
                           description=self.DESCRIPTION)
        self.app.get("/health")(self.health)
        self.app.get("/search")(self.search)
        self.app.get("/answer")(self.answer)

    def health(self) -> dict[str, str]:
        """Check that the server is up (it never loads anything)."""
        return {"status": "ok"}

    def search(self,
               query: Annotated[str, Query(min_length=1)],
               k: Annotated[int, Query(gt=0)] = 10
               ) -> StudentSearchResults:
        """Return the ``k`` best source locations for a question.

        Same JSON as the ``search`` command, without saving a file.
        """
        question = UnansweredQuestion(question=query)
        with self._retriever_lock:
            results = StudentSearchResults(
                search_results=[RagCLI._search_question(
                    self._retriever,
                    question,
                    k
                    )],
                k=k
            )
            self._retriever.flush()
        return results

    def _get_generator(self) -> "Generator":
        """Return the generator, loading it on the first call.

        Must be called with ``_generator_lock`` held.

        Raises:
            HTTPException: 503 when PyTorch cannot be imported or the
                model does not fit in memory; the next request tries
                again.
        """
        if self._generator is None:
            try:
                print("Loading PyTorch...", file=sys.stderr)
                from .generator import Generator
            except (ImportError, MemoryError, OSError) as e:
                raise HTTPException(
                    status_code=503,
                    detail=f"Cannot load PyTorch ({type(e).__name__}: {e})."
                ) from e
            print(f"Loading {Generator.MODEL_NAME}...", file=sys.stderr)
            try:
                self._generator = Generator(self._dtype)
            except RagError as e:
                raise HTTPException(status_code=503, detail=str(e)) from e
        return self._generator

    def answer(self,
               query: Annotated[str, Query(min_length=1)],
               k: Annotated[int, Query(gt=0)] = 10
               ) -> StudentSearchResultsAndAnswer:
        """Answer a question from its ``k`` best sources.

        Same JSON as the ``answer`` command, without saving a file. When
        no source is found, the answer is a fixed message and the model
        is not loaded.
        """
        question = UnansweredQuestion(question=query)
        with self._retriever_lock:
            hits = self._retriever.search(query, k)
            self._retriever.flush()
        # No source found: nothing to ground an answer on, so the model
        # is not loaded.
        if hits:
            with self._generator_lock:
                answer = self._get_generator().generate(
                    query, [hit.chunk for hit in hits])
        else:
            answer = NO_SOURCES
        answered = MinimalAnswer(
            question_id=question.question_id,
            question=query,
            retrieved_sources=[MinimalSource(
                file_path=hit.chunk.file_path,
                first_character_index=hit.chunk.first_character_index,
                last_character_index=hit.chunk.last_character_index
                ) for hit in hits],
            answer=answer
        )
        return StudentSearchResultsAndAnswer(search_results=[answered], k=k)
