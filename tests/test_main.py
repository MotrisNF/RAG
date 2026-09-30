"""Tests for the entry point ``src.__main__.main``.

``main`` reads the command from ``sys.argv``; the tests replace it and
check the exit code and what is printed on stderr.
"""
import sys
from pathlib import Path

import fire
import pytest

from src.__main__ import main
from src.exceptions import CorpusError
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
    def test_exit_zero(self, runner: Runner) -> None:
        """Exit code 0 and no error message (stderr may hold tqdm bars)."""
        code, err = runner.run("index", "--raw_dir", str(REAL_RAW))
        assert code == 0
        for word in ("Error", "error", "Invalid", "Traceback"):
            assert word not in err


class TestPendingCommands:
    """Commands not written yet fail loudly and cleanly.

    When a command gets implemented, remove it from this list.
    """

    @pytest.mark.parametrize("args", [
        ("search", "How to use LoRA?", "--k", "5"),
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
    """The ``index`` command (for now: read the corpus and summarize)."""

    @pytest.mark.skipif(not REAL_RAW.is_dir(),
                        reason="vLLM corpus not present")
    def test_summary(self, runner: Runner) -> None:
        """It reports how many files were read and of which types."""
        code, out, err = runner.run_full("index", "--raw_dir", str(REAL_RAW))
        assert code == 0
        assert out.startswith("Read ")
        assert ".md:" in out and ".py:" in out and ".txt:" in out
        assert "Skipped" not in out
        assert "Reading corpus" in err

    @pytest.mark.skipif(not REAL_RAW.is_dir(),
                        reason="vLLM corpus not present")
    def test_chunk_size_received(self, runner: Runner) -> None:
        """``--max_chunk_size`` reaches the command."""
        _, out, _ = runner.run_full("index", "--raw_dir", str(REAL_RAW),
                                    "--max_chunk_size", "500")
        assert "max_chunk_size = 500" in out

    def test_custom_raw_dir(self, runner: Runner,
                            monkeypatch: pytest.MonkeyPatch,
                            tmp_path: Path) -> None:
        """``--raw_dir`` makes the command read another folder."""
        monkeypatch.setattr(CorpusReader, "PROJECT_ROOT", tmp_path)
        corpus = tmp_path / "my_corpus"
        corpus.mkdir()
        (corpus / "a.md").write_text("hello", encoding="utf-8")
        (corpus / "b.py").write_text("x = 1", encoding="utf-8")
        code, out, _ = runner.run_full("index", "--raw_dir", str(corpus))
        assert code == 0
        assert out.startswith("Read 2 of 2 files (10 characters)")
        assert ".md: 1 | .py: 1" in out

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
        assert "at least 1 character" in err

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
