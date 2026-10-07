"""Tests for ``src.api.RagApi`` (bonus 5), through FastAPI's TestClient.

No test loads PyTorch or the real model: ``/answer`` gets a fake
generator, and the tests of a failed load replace ``src.generator`` in
``sys.modules`` before the endpoint imports it.
"""
import sys
import threading
import time
from pathlib import Path
from types import ModuleType

import pytest
from fastapi.testclient import TestClient

from src.api import RagApi
from src.base_retriever import Retriever
from src.bm25_index import BM25Index
from src.cached_retriever import CachedRetriever
from src.exceptions import RagError
from src.models import (NO_SOURCES, Chunk, SearchHit, StudentSearchResults,
                        StudentSearchResultsAndAnswer)

TEXTS = ["Load a LoRA adapter at runtime with the API.",
         "Configure the OpenAI compatible server.",
         "LoRA adapters can be served per request."]


class FakeGenerator:
    """Stand-in for ``Generator``: records the calls, answers at once."""

    def __init__(self) -> None:
        """No call yet."""
        self.calls: list[tuple[str, list[Chunk]]] = []

    def generate(self, question: str, chunks: list[Chunk]) -> str:
        """Record the call; the answer names the number of chunks."""
        self.calls.append((question, chunks))
        return f"Answer from {len(chunks)} chunks."


class Empty(Retriever):
    """Retriever that never finds anything."""

    def search(self, query: str, k: int) -> list[SearchHit]:
        """No hits."""
        return []


@pytest.fixture
def bm25(capsys: pytest.CaptureFixture[str]) -> BM25Index:
    """A small BM25 index in memory (its progress bar is discarded)."""
    chunks = [Chunk(file_path=f"data/raw/{i}.md",
                    first_character_index=0,
                    last_character_index=len(text), text=text)
              for i, text in enumerate(TEXTS)]
    index = BM25Index.build(chunks)
    capsys.readouterr()
    return index


@pytest.fixture
def api(bm25: BM25Index) -> RagApi:
    """The API over the small index, with a fake model already loaded."""
    rag = RagApi(bm25)
    rag._generator = FakeGenerator()  # type: ignore[assignment]
    return rag


@pytest.fixture
def client(api: RagApi) -> TestClient:
    """HTTP client for ``api``."""
    return TestClient(api.app)


def fake_generator_module(monkeypatch: pytest.MonkeyPatch,
                          generator_class: type | None) -> None:
    """Make ``from .generator import Generator`` give ``generator_class``
    (``None``: the import fails, as without PyTorch)."""
    if generator_class is None:
        monkeypatch.setitem(sys.modules, "src.generator", None)
        return
    module = ModuleType("src.generator")
    module.Generator = generator_class  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "src.generator", module)


class TestHealth:
    """``/health``."""

    def test_ok(self, client: TestClient) -> None:
        """Always ``{"status": "ok"}``."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestSearch:
    """``/search``."""

    def test_valid_results(self, client: TestClient) -> None:
        """A ``StudentSearchResults`` with the question and its sources."""
        response = client.get("/search",
                              params={"query": "lora adapter", "k": 2})
        assert response.status_code == 200
        results = StudentSearchResults.model_validate(response.json())
        assert results.k == 2
        [found] = results.search_results
        assert found.question == "lora adapter"
        assert len(found.retrieved_sources) == 2
        assert found.retrieved_sources[0].file_path in (
            "data/raw/0.md", "data/raw/2.md")

    def test_default_k(self, client: TestClient) -> None:
        """Without ``k``, 10 (the CLI default)."""
        response = client.get("/search", params={"query": "lora"})
        assert response.json()["k"] == 10

    def test_no_text_in_output(self, client: TestClient) -> None:
        """Only locations: no chunk text and no score."""
        response = client.get("/search", params={"query": "lora"})
        source = response.json()["search_results"][0]["retrieved_sources"][0]
        assert set(source) == {"file_path", "first_character_index",
                               "last_character_index"}

    def test_new_id_each_request(self, client: TestClient) -> None:
        """Every request is a new question, with its own id."""
        ids = {client.get("/search", params={"query": "lora"}).json()
               ["search_results"][0]["question_id"] for _ in range(2)}
        assert len(ids) == 2


class TestValidation:
    """Invalid parameters give 422, on both endpoints."""

    @pytest.mark.parametrize("endpoint", ["/search", "/answer"])
    @pytest.mark.parametrize("params", [
        {},
        {"query": ""},
        {"query": "lora", "k": 0},
        {"query": "lora", "k": -1},
        {"query": "lora", "k": "abc"},
        {"query": "lora", "k": 1.5},
    ])
    def test_rejected(self, client: TestClient, api: RagApi, endpoint: str,
                      params: dict[str, str | int | float]) -> None:
        """422 and the model is never called."""
        response = client.get(endpoint, params=params)
        assert response.status_code == 422
        assert api._generator.calls == []  # type: ignore[union-attr]

    def test_unknown_path(self, client: TestClient) -> None:
        """404 for an endpoint that does not exist."""
        assert client.get("/nope").status_code == 404


class TestAnswer:
    """``/answer`` with the fake model."""

    def test_valid_answer(self, client: TestClient, api: RagApi) -> None:
        """A ``StudentSearchResultsAndAnswer`` with the generated text."""
        response = client.get("/answer",
                              params={"query": "lora adapter", "k": 2})
        assert response.status_code == 200
        results = StudentSearchResultsAndAnswer.model_validate(
            response.json())
        assert results.k == 2
        [answered] = results.search_results
        assert answered.question == "lora adapter"
        assert answered.answer == "Answer from 2 chunks."
        assert len(answered.retrieved_sources) == 2

    def test_model_gets_chunk_texts(self, client: TestClient,
                                    api: RagApi) -> None:
        """The model receives the question and the full chunks, in the
        same order as the sources."""
        response = client.get("/answer",
                              params={"query": "lora adapter", "k": 2})
        [(question, chunks)] = api._generator.calls  # type: ignore
        assert question == "lora adapter"
        assert all(chunk.text in TEXTS for chunk in chunks)
        sources = response.json()["search_results"][0]["retrieved_sources"]
        assert [c.file_path for c in chunks] == [
            s["file_path"] for s in sources]

    def test_no_sources(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No hit: the fixed message, and the model is not loaded."""
        fake_generator_module(monkeypatch, None)
        rag = RagApi(Empty())
        response = TestClient(rag.app).get("/answer",
                                           params={"query": "lora"})
        assert response.status_code == 200
        answered = response.json()["search_results"][0]
        assert answered["answer"] == NO_SOURCES
        assert answered["retrieved_sources"] == []
        assert rag._generator is None


class TestModelLoading:
    """The model: loaded once, on the first ``/answer``."""

    def test_not_loaded_at_start(self, bm25: BM25Index) -> None:
        """Creating the API and searching never load it."""
        rag = RagApi(bm25)
        TestClient(rag.app).get("/search", params={"query": "lora"})
        assert rag._generator is None

    def test_loaded_once_with_dtype(self, bm25: BM25Index,
                                    monkeypatch: pytest.MonkeyPatch) -> None:
        """Several answers, one load, with the server's dtype."""
        dtypes: list[str] = []

        class Counting(FakeGenerator):
            """Records the dtype of each load."""

            MODEL_NAME = "fake"

            def __init__(self, dtype: str) -> None:
                """Record the load."""
                super().__init__()
                dtypes.append(dtype)

        fake_generator_module(monkeypatch, Counting)
        client = TestClient(RagApi(bm25, "bfloat16").app)
        for _ in range(3):
            assert client.get("/answer",
                              params={"query": "lora"}).status_code == 200
        assert dtypes == ["bfloat16"]

    def test_no_pytorch_503(self, bm25: BM25Index,
                            monkeypatch: pytest.MonkeyPatch) -> None:
        """PyTorch cannot be imported: 503 with the reason."""
        fake_generator_module(monkeypatch, None)
        response = TestClient(RagApi(bm25).app).get(
            "/answer", params={"query": "lora"})
        assert response.status_code == 503
        assert "Cannot load PyTorch" in response.json()["detail"]

    def test_not_enough_memory_503(self, bm25: BM25Index,
                                   monkeypatch: pytest.MonkeyPatch) -> None:
        """The model refuses to load (RagError): 503 with its message,
        and the server keeps answering ``/search``."""

        class NoMemory:
            """Fails like ``Generator`` without enough memory."""

            MODEL_NAME = "fake"

            def __init__(self, dtype: str) -> None:
                """Always refuse."""
                raise RagError("Not enough memory to load the model.")

        fake_generator_module(monkeypatch, NoMemory)
        rag = RagApi(bm25)
        client = TestClient(rag.app)
        response = client.get("/answer", params={"query": "lora"})
        assert response.status_code == 503
        assert response.json() == {
            "detail": "Not enough memory to load the model."}
        assert rag._generator is None
        assert client.get("/search",
                          params={"query": "lora"}).status_code == 200


class TestConcurrency:
    """The locks: requests at the same time do not run together."""

    def test_one_search_at_a_time(self) -> None:
        """Two searches in parallel never overlap inside the retriever."""

        class Slow(Retriever):
            """Records how many searches run at the same time."""

            def __init__(self) -> None:
                """Nothing running yet."""
                self.running = 0
                self.most = 0

            def search(self, query: str, k: int) -> list[SearchHit]:
                """Wait a little while counted as running."""
                self.running += 1
                self.most = max(self.most, self.running)
                time.sleep(0.05)
                self.running -= 1
                return []

        retriever = Slow()
        client = TestClient(RagApi(retriever).app)
        threads = [threading.Thread(
            target=client.get, args=("/search",),
            kwargs={"params": {"query": "lora"}}) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert retriever.most == 1


class TestCache:
    """With a ``CachedRetriever``, every request saves the cache."""

    def test_saved_after_request(self, bm25: BM25Index,
                                 tmp_path: Path) -> None:
        """A new question is in the cache file after ``/search``."""
        path = tmp_path / "query_cache_bm25.json"
        cached = CachedRetriever(lambda: bm25, "test", path)
        TestClient(RagApi(cached).app).get(
            "/search", params={"query": "lora", "k": 2})
        assert "2:lora" in path.read_text(encoding="utf-8")
