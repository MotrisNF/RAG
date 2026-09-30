"""Tests for the entry point ``src.__main__.main``.

``main`` reads the command from ``sys.argv``; the tests replace it and
check the exit code and what is printed on stderr.
"""
import json
import sys
import uuid
from pathlib import Path

import fire
import pytest

from src.__main__ import main
from src.bm25_index import BM25Index
from src.exceptions import CorpusError
from src.models import Chunk, StudentSearchResults
from src.reader import CorpusReader

DOCS_DATASET = "data/datasets/UnansweredQuestions/dataset_docs_public.json"
REAL_RAW = CorpusReader.PROJECT_ROOT / "data" / "raw"


class Runner:
    """Run ``main`` with a given command line."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch,
                 capsys: pytest.CaptureFixture[str]) -> None:
        """Keep the pytest helpers.

        Args:
            monkeypatch: Used to replace ``sys.argv``.
            capsys: Used to read stdout and stderr.
        """
        self._monkeypatch = monkeypatch
        self._capsys = capsys

    def run(self, *args: str) -> tuple[int, str]:
        """Run ``main`` as ``python -m src <args>``.

        Returns:
            The exit code (0 when ``main`` returns normally) and stderr.
        """
        code, _, err = self.run_full(*args)
        return code, err

    def run_full(self, *args: str) -> tuple[int, str, str]:
        """Like ``run`` but also return stdout.

        Returns:
            The exit code, stdout and stderr.
        """
        self._monkeypatch.setattr(sys, "argv", ["src", *args])
        code = 0
        try:
            main()
        except SystemExit as exit_:
            code = exit_.code if isinstance(exit_.code, int) else 1
        captured = self._capsys.readouterr()
        return code, captured.out, captured.err

    def run_raising(self, error: BaseException) -> tuple[int, str]:
        """Run ``main`` with Fire replaced by a function that raises."""
        def raiser(*args: object, **kwargs: object) -> None:
            raise error
        self._monkeypatch.setattr(fire, "Fire", raiser)
        return self.run("search", "x")


@pytest.fixture
def runner(monkeypatch: pytest.MonkeyPatch,
           capsys: pytest.CaptureFixture[str]) -> Runner:
    """A ``Runner`` bound to this test's helpers."""
    return Runner(monkeypatch, capsys)


class TestSuccess:
    """Valid commands finish without exiting with an error."""

    @pytest.mark.skipif(not REAL_RAW.is_dir(),
                        reason="vLLM corpus not present")
    def test_exit_zero(self, runner: Runner, tmp_path: Path) -> None:
        """Exit code 0 and no error message (stderr may hold tqdm bars)."""
        code, err = runner.run("index", "--raw_dir", str(REAL_RAW),
                               "--index_dir", str(tmp_path))
        assert code == 0
        for word in ("Error", "error", "Invalid", "Traceback"):
            assert word not in err


class TestSearch:
    """The ``search`` command: one question, JSON output on stdout."""

    TEXTS = ["Load a LoRA adapter at runtime with the API.",
             "Configure the OpenAI compatible server.",
             "LoRA adapters can be served per request.",
             "Unrelated text about tokenizers."]

    @pytest.fixture
    def index_dir(self, tmp_path: Path,
                  capsys: pytest.CaptureFixture[str]) -> Path:
        """A small saved index (its build progress bar is discarded)."""
        chunks = [Chunk(file_path=f"data/raw/{i}.md",
                        first_character_index=10 * i,
                        last_character_index=10 * i + len(text), text=text)
                  for i, text in enumerate(self.TEXTS)]
        BM25Index.build(chunks).save(tmp_path)
        capsys.readouterr()
        return tmp_path

    def search(self, runner: Runner, index_dir: Path,
               *args: str) -> tuple[int, StudentSearchResults, str]:
        """Run ``search`` and parse its stdout as StudentSearchResults."""
        code, out, err = runner.run_full("search", *args,
                                         "--index_dir", str(index_dir))
        return code, StudentSearchResults.model_validate_json(out), err

    def test_output_is_student_search_results(self, runner: Runner,
                                              index_dir: Path) -> None:
        """stdout is exactly one valid StudentSearchResults JSON."""
        code, results, err = self.search(runner, index_dir, "lora adapter",
                                         "--k", "2")
        assert code == 0
        assert err == ""
        assert results.k == 2
        [entry] = results.search_results
        assert entry.question == "lora adapter"
        assert [s.file_path for s in entry.retrieved_sources] == [
            "data/raw/0.md", "data/raw/2.md"]

    def test_sources_keep_character_range(self, runner: Runner,
                                          index_dir: Path) -> None:
        """Each source carries the chunk's exact character range."""
        _, results, _ = self.search(runner, index_dir, "openai server")
        source = results.search_results[0].retrieved_sources[0]
        assert (source.file_path, source.first_character_index,
                source.last_character_index) == (
            "data/raw/1.md", 10, 10 + len(self.TEXTS[1]))

    def test_no_chunk_text_in_output(self, runner: Runner,
                                     index_dir: Path) -> None:
        """Only the subject's fields: no text, no score."""
        code, out, _ = runner.run_full("search", "lora",
                                       "--index_dir", str(index_dir))
        source = json.loads(out)["search_results"][0]["retrieved_sources"][0]
        assert set(source) == {"file_path", "first_character_index",
                               "last_character_index"}
        assert "Load a LoRA adapter" not in out

    def test_question_id_is_uuid(self, runner: Runner,
                                 index_dir: Path) -> None:
        """A single query gets a generated id, like a dataset question."""
        _, results, _ = self.search(runner, index_dir, "lora")
        uuid.UUID(results.search_results[0].question_id)

    def test_k_limits_sources(self, runner: Runner, index_dir: Path) -> None:
        """At most k sources, fewer if fewer chunks match."""
        _, one, _ = self.search(runner, index_dir, "lora", "--k", "1")
        _, many, _ = self.search(runner, index_dir, "lora", "--k", "50")
        assert len(one.search_results[0].retrieved_sources) == 1
        assert len(many.search_results[0].retrieved_sources) == 2
        assert many.k == 50

    def test_no_match(self, runner: Runner, index_dir: Path) -> None:
        """Nothing matches: valid JSON with an empty source list."""
        code, results, _ = self.search(runner, index_dir, "zzzqqq")
        assert code == 0
        assert results.search_results[0].retrieved_sources == []

    def test_missing_index(self, runner: Runner, tmp_path: Path) -> None:
        """No index yet: clear message, exit 1, nothing on stdout."""
        code, out, err = runner.run_full("search", "lora", "--index_dir",
                                         str(tmp_path / "none"))
        assert code == 1
        assert out == ""
        assert "Run 'index' first" in err

    def test_empty_query_rejected(self, runner: Runner,
                                  index_dir: Path) -> None:
        """Validation still runs before loading anything."""
        code, err = runner.run("search", "  ", "--index_dir", str(index_dir))
        assert code == 1
        assert "--query '  ': must not be empty or blank" in err


class TestPendingCommands:
    """Commands not written yet fail loudly and cleanly.

    When a command gets implemented, remove it from this list.
    """

    @pytest.mark.parametrize("args", [
        ("search_dataset", "--dataset_path", DOCS_DATASET),
        ("answer", "How to use LoRA?"),
        ("answer_dataset", "--student_search_results_path", DOCS_DATASET),
        ("evaluate", "--student_search_results_path", DOCS_DATASET,
         "--dataset_path", DOCS_DATASET),
    ])
    def test_not_implemented(self, runner: Runner,
                             args: tuple[str, ...]) -> None:
        """Exit 1 with ``NotImplementedError`` and no traceback."""
        code, err = runner.run(*args)
        assert code == 1
        assert "NotImplementedError" in err
        assert f"{args[0]} is not implemented yet" in err
        assert "Traceback" not in err


class TestIndex:
    """The ``index`` command: read, chunk, build and save the index.

    Every test saves into ``tmp_path``, never into ``data/processed``.
    """

    @pytest.mark.skipif(not REAL_RAW.is_dir(),
                        reason="vLLM corpus not present")
    def test_summary(self, runner: Runner, tmp_path: Path) -> None:
        """It reports how many files were read and of which types."""
        code, out, err = runner.run_full("index", "--raw_dir", str(REAL_RAW),
                                         "--index_dir", str(tmp_path))
        assert code == 0
        assert out.startswith("Read ")
        assert ".md:" in out and ".py:" in out and ".txt:" in out
        assert "Skipped" not in out
        assert "Reading corpus" in err

    @pytest.mark.skipif(not REAL_RAW.is_dir(),
                        reason="vLLM corpus not present")
    def test_chunk_size_received(self, runner: Runner,
                                 tmp_path: Path) -> None:
        """``--max_chunk_size`` reaches the command."""
        _, out, _ = runner.run_full("index", "--raw_dir", str(REAL_RAW),
                                    "--max_chunk_size", "500",
                                    "--index_dir", str(tmp_path))
        assert "(max 500 characters)" in out

    def test_custom_raw_dir(self, runner: Runner,
                            monkeypatch: pytest.MonkeyPatch,
                            tmp_path: Path) -> None:
        """``--raw_dir`` makes the command read another folder."""
        monkeypatch.setattr(CorpusReader, "PROJECT_ROOT", tmp_path)
        corpus = tmp_path / "my_corpus"
        corpus.mkdir()
        (corpus / "a.md").write_text("hello", encoding="utf-8")
        (corpus / "b.py").write_text("x = 1", encoding="utf-8")
        code, out, _ = runner.run_full("index", "--raw_dir", str(corpus),
                                       "--index_dir", str(tmp_path / "idx"))
        assert code == 0
        assert out.startswith("Read 2 of 2 files (10 characters)")
        assert ".md: 1 | .py: 1" in out

    def test_saves_index(self, runner: Runner,
                         monkeypatch: pytest.MonkeyPatch,
                         tmp_path: Path) -> None:
        """The index file is written into ``--index_dir`` (created if
        missing) and the final message says where.
        """
        monkeypatch.setattr(CorpusReader, "PROJECT_ROOT", tmp_path)
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "a.md").write_text("load a lora adapter", encoding="utf-8")
        index_dir = tmp_path / "new" / "index"
        code, out, _ = runner.run_full("index", "--raw_dir", str(corpus),
                                       "--index_dir", str(index_dir))
        assert code == 0
        assert (index_dir / BM25Index.FILE_NAME).is_file()
        assert f"Ingestion complete! Index saved under {index_dir}/" in out
        hits = BM25Index.load(index_dir).search("lora", 1)
        assert hits[0].chunk.file_path == "corpus/a.md"

    def test_index_dir_is_a_file(self, runner: Runner,
                                 monkeypatch: pytest.MonkeyPatch,
                                 tmp_path: Path) -> None:
        """``--index_dir`` pointing to a file: clear message, exit 1."""
        monkeypatch.setattr(CorpusReader, "PROJECT_ROOT", tmp_path)
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "a.md").write_text("text", encoding="utf-8")
        not_a_dir = tmp_path / "file.txt"
        not_a_dir.write_text("x", encoding="utf-8")
        code, err = runner.run("index", "--raw_dir", str(corpus),
                               "--index_dir", str(not_a_dir))
        assert code == 1
        assert f"Cannot use '{not_a_dir}'" in err
        assert "Traceback" not in err

    def test_empty_index_dir(self, runner: Runner) -> None:
        """An empty ``--index_dir`` is rejected before reading anything."""
        code, err = runner.run("index", "--index_dir", "")
        assert code == 1
        assert "--index_dir '': must not be empty or blank" in err

    def test_missing_corpus(self, runner: Runner, tmp_path: Path) -> None:
        """A corpus folder that does not exist: clear message and exit 1."""
        code, err = runner.run("index", "--raw_dir", str(tmp_path / "no"))
        assert code == 1
        assert "Corpus folder not found" in err
        assert "Traceback" not in err

    def test_empty_raw_dir(self, runner: Runner) -> None:
        """An empty ``--raw_dir`` is rejected before reading anything."""
        code, err = runner.run("index", "--raw_dir", "")
        assert code == 1
        assert "Invalid arguments:" in err


class TestArgumentErrors:
    """Invalid values become ``ArgError``: message and exit code 1."""

    def test_k_zero(self, runner: Runner) -> None:
        """``--k 0`` is rejected with a readable message."""
        code, err = runner.run("search", "hola", "--k", "0")
        assert code == 1
        assert "Invalid arguments:" in err
        assert "'0': Input should be greater than 0" in err

    def test_empty_query(self, runner: Runner) -> None:
        """An empty query is rejected."""
        code, err = runner.run("search", "")
        assert code == 1
        assert "--query '': must not be empty or blank" in err

    def test_missing_file(self, runner: Runner) -> None:
        """A dataset that does not exist is rejected."""
        code, err = runner.run("search_dataset", "--dataset_path", "no.json")
        assert code == 1
        assert "'no.json'" in err

    def test_no_traceback(self, runner: Runner) -> None:
        """The user never sees a Python traceback."""
        _, err = runner.run("search", "hola", "--k", "abc")
        assert "Traceback" not in err


class TestUsageErrors:
    """Fire handles missing arguments and unknown commands: exit 2."""

    @pytest.mark.parametrize("args", [
        ("search",),
        ("not_a_command",),
    ])
    def test_exit_two(self, runner: Runner, args: tuple[str, ...]) -> None:
        """Fire's usage error exits with code 2."""
        code, _ = runner.run(*args)
        assert code == 2


class TestCommandNeverRuns:
    """Malformed command lines stop BEFORE the command runs.

    ``CorpusReader.read`` is replaced by a spy: ``index`` must not touch
    the corpus in any of these cases.
    """

    @pytest.fixture
    def reads(self, monkeypatch: pytest.MonkeyPatch) -> list[str]:
        """Record every call to ``CorpusReader.read``."""
        calls: list[str] = []

        def spy(self: CorpusReader) -> None:
            calls.append("read")
            raise AssertionError("the corpus was read")

        monkeypatch.setattr(CorpusReader, "read", spy)
        return calls

    def test_unknown_option(self, runner: Runner, reads: list[str]) -> None:
        """A typo in an option: exit 2, clear message, nothing read."""
        code, err = runner.run("index", "--chunk_size", "5")
        assert code == 2
        assert "Unknown option '--chunk_size' for 'index'" in err
        assert "--max_chunk_size" in err
        assert reads == []

    def test_missing_value(self, runner: Runner, reads: list[str]) -> None:
        """``--raw_dir`` with no value: exit 2, clear message, nothing read.

        Without the check Fire would pass ``raw_dir=True``.
        """
        code, err = runner.run("index", "--max_chunk_size", "10",
                               "--raw_dir")
        assert code == 2
        assert "Option '--raw_dir' of 'index' needs a value" in err
        assert "True" not in err
        assert reads == []

    def test_too_many_arguments(self, runner: Runner,
                                reads: list[str]) -> None:
        """Extra values: exit 2, nothing read."""
        code, err = runner.run("index", "1", "data/raw", "data/processed",
                               "extra")
        assert code == 2
        assert "Too many arguments for 'index'" in err
        assert reads == []

    @pytest.mark.parametrize("args", [
        ("index", "--max_chunk_size", "5", "--help"),
        ("index", "-h"),
        ("index", "--help"),
    ])
    def test_help(self, runner: Runner, reads: list[str],
                  args: tuple[str, ...]) -> None:
        """Help is shown (Fire writes it to stderr) and nothing runs."""
        code, _, err = runner.run_full(*args)
        assert code == 0
        assert "max_chunk_size" in err
        assert reads == []

    def test_valid_forms_still_run(self, runner: Runner,
                                   reads: list[str]) -> None:
        """Fire's alternative spellings reach the command."""
        for args in (("index", "--max-chunk-size", "5"),
                     ("index", "-m", "5"),
                     ("index", "--max_chunk_size=5")):
            code, _ = runner.run(*args)
            assert code == 1
        assert reads == ["read", "read", "read"]


class TestErrorHandling:
    """Errors raised inside a command."""

    def test_rag_error(self, runner: Runner) -> None:
        """Project errors: message on stderr, exit 1."""
        code, err = runner.run_raising(CorpusError("Corpus folder not found"))
        assert code == 1
        assert err.strip() == "Corpus folder not found"

    def test_keyboard_interrupt(self, runner: Runner) -> None:
        """Ctrl+C: clean message, exit 130."""
        code, err = runner.run_raising(KeyboardInterrupt())
        assert code == 130
        assert "Interrupted by user." in err

    def test_unexpected_error(self, runner: Runner) -> None:
        """Any other error shows its type, exit 1, no traceback."""
        code, err = runner.run_raising(KeyError("rag_questions"))
        assert code == 1
        assert "Unexpected error: KeyError: 'rag_questions'" in err
        assert "Traceback" not in err

    def test_error_without_message(self, runner: Runner) -> None:
        """The type name is shown even when the message is empty."""
        _, err = runner.run_raising(RuntimeError())
        assert "Unexpected error: RuntimeError" in err

    def test_system_exit_not_swallowed(self, runner: Runner) -> None:
        """``SystemExit`` is not treated as an unexpected error."""
        code, err = runner.run_raising(SystemExit(3))
        assert code == 3
        assert "Unexpected error" not in err
