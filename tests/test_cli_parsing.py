"""Tests for the command-line argument parsing of ``RagCLI``.

Each test calls Fire exactly as the terminal would, passing the arguments
as a list. They only check parsing and validation, not what each command
does.
"""
from pathlib import Path

import fire
import pytest
from src.bm25_index import BM25Index
from src.commands import RagCLI
from src.exceptions import ArgError
from src.models import Chunk

DOCS_DATASET = "data/datasets/UnansweredQuestions/dataset_docs_public.json"
DOCS_ANSWERED = "data/datasets/AnsweredQuestions/dataset_docs_public.json"
MISSING_FILE = "does_not_exist.json"
A_DIRECTORY = "data"


class TestValidArguments:
    """Arguments that must be accepted without raising."""

    @pytest.mark.parametrize("command", [
        ["index"],
        ["index", "--max_chunk_size", "500"],
        ["index", "--raw_dir", "data/raw"],
        ["search", "How to use LoRA?"],
        ["search", "How to use LoRA?", "--k", "5"],
        ["search", "How to use LoRA?", "--k=5"],
        ["search", "123"],
        ["search_dataset", "--dataset_path", DOCS_DATASET],
        ["search_dataset", "--dataset_path", DOCS_DATASET, "--k", "10",
         "--save_directory", "folder/that/does/not/exist/yet"],
        ["answer", "How to use LoRA?", "--k", "3"],
        ["answer_dataset", "--student_search_results_path", DOCS_DATASET],
        ["answer_dataset", "--student_search_results_path", DOCS_DATASET,
         "--save_directory", "folder/that/does/not/exist/yet"],
        ["evaluate", "--student_search_results_path", DOCS_DATASET,
         "--dataset_path", DOCS_ANSWERED],
        ["evaluate", "--student_search_results_path", DOCS_DATASET,
         "--dataset_path", DOCS_ANSWERED, "--k", "5"],
    ])
    def test_accepted(self, command: list[str], tmp_path: Path) -> None:
        """Fire runs the command and pydantic accepts every value.

        Commands not written yet raise ``NotImplementedError`` from their
        body: reaching the body means validation already passed. ``index``
        saves into a temporary folder, never into ``data/processed``, and
        ``search_dataset`` writes its results into a temporary folder too.
        """
        command = [str(tmp_path / arg) if arg.startswith("folder/") else arg
                   for arg in command]
        if command[0] == "index":
            command = [*command, "--index_dir", str(tmp_path)]
        if command[0] == "search_dataset":
            if "--save_directory" not in command:
                command = [*command, "--save_directory", str(tmp_path)]
        if command[0] in ("search", "search_dataset"):
            chunk = Chunk(file_path="data/raw/a.md", first_character_index=0,
                          last_character_index=15, text="How to use LoRA")
            BM25Index.build([chunk]).save(tmp_path)
            command = [*command, "--index_dir", str(tmp_path)]
        try:
            fire.Fire(RagCLI, command=command)
        except NotImplementedError:
            pass


class TestInvalidValues:
    """Values of the right argument but wrong content.

    pydantic rejects them and the command raises ``ArgError``. Any other
    exception (for example ``TypeError``) makes the test fail.
    """

    @pytest.mark.parametrize("command", [
        # Empty or blank queries
        ["search", ""],
        ["search", "   "],
        ["answer", ""],
        # Invalid k
        ["search", "hola", "--k", "0"],
        ["search", "hola", "--k", "-3"],
        ["search", "hola", "--k", "abc"],
        ["search", "hola", "--k", "3.7"],
        ["answer", "hola", "--k", "0"],
        ["search_dataset", "--dataset_path", DOCS_DATASET, "--k", "0"],
        ["evaluate", "--student_search_results_path", DOCS_DATASET,
         "--dataset_path", DOCS_ANSWERED, "--k", "0"],
        # Invalid max_chunk_size
        ["index", "--max_chunk_size", "0"],
        ["index", "--max_chunk_size", "-100"],
        ["index", "--max_chunk_size", "big"],
        # Input files that do not exist or are directories
        ["search_dataset", "--dataset_path", MISSING_FILE],
        ["search_dataset", "--dataset_path", A_DIRECTORY],
        ["answer_dataset", "--student_search_results_path", MISSING_FILE],
        ["evaluate", "--student_search_results_path", MISSING_FILE,
         "--dataset_path", DOCS_ANSWERED],
        ["evaluate", "--student_search_results_path", DOCS_DATASET,
         "--dataset_path", MISSING_FILE],
        # Empty directories
        ["search_dataset", "--dataset_path", DOCS_DATASET,
         "--save_directory", ""],
        ["index", "--raw_dir", ""],
        ["index", "--raw_dir", "   "],
    ])
    def test_rejected(self, command: list[str]) -> None:
        """The command raises ``ArgError``."""
        with pytest.raises(ArgError):
            fire.Fire(RagCLI, command=command)


class TestWrongUsage:
    """Missing or unknown arguments.

    Fire prints the usage and exits with code 2 (``FireExit``, a
    ``SystemExit``) before pydantic is reached.
    """

    @pytest.mark.parametrize("command", [
        ["search"],
        ["answer"],
        ["answer_dataset"],
        ["evaluate", "--dataset_path", DOCS_ANSWERED],
        ["not_a_command"],
    ])
    def test_usage_error(self, command: list[str]) -> None:
        """Fire exits with a usage error."""
        with pytest.raises(SystemExit):
            fire.Fire(RagCLI, command=command)


class TestUnknownFlag:
    """Known Fire behaviour: an unknown flag is reported only AFTER the
    command has run (Fire calls the method with the arguments it
    understood, then fails on the leftover ones).
    """

    def test_command_runs_before_usage_error(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The method is called once, then Fire exits with code 2."""
        calls: list[tuple[str, int]] = []

        def spy(self: RagCLI, query: str, k: int = 10) -> None:
            calls.append((query, k))

        monkeypatch.setattr(RagCLI, "search", spy)
        with pytest.raises(SystemExit) as info:
            fire.Fire(RagCLI, command=["search", "hola", "--unknown", "5"])
        assert info.value.code == 2
        assert calls == [("hola", 10)]
