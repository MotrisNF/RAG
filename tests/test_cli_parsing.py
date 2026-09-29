"""Tests for the command-line argument parsing of ``RagCLI``.

Each test calls Fire exactly as the terminal would, passing the arguments
as a list. They only check parsing and validation, not what each command
does.
"""
import fire
import pytest
from pydantic import ValidationError

from src.commands import RagCLI

DOCS_DATASET = "data/datasets/UnansweredQuestions/dataset_docs_public.json"
DOCS_ANSWERED = "data/datasets/AnsweredQuestions/dataset_docs_public.json"
MISSING_FILE = "does_not_exist.json"
A_DIRECTORY = "data"


class TestValidArguments:
    """Arguments that must be accepted without raising."""

    @pytest.mark.parametrize("command", [
        ["index"],
        ["index", "--max_chunk_size", "500"],
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
    def test_accepted(self, command: list[str]) -> None:
        """Fire runs the command and pydantic accepts every value."""
        fire.Fire(RagCLI, command=command)


class TestInvalidValues:
    """Values of the right argument but wrong content.

    pydantic must reject them with a ``ValidationError``. Any other
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
        # Empty output directory
        ["search_dataset", "--dataset_path", DOCS_DATASET,
         "--save_directory", ""],
    ])
    def test_rejected(self, command: list[str]) -> None:
        """pydantic raises ``ValidationError``."""
        with pytest.raises(ValidationError):
            fire.Fire(RagCLI, command=command)


class TestWrongUsage:
    """Missing or unknown arguments.

    Fire prints the usage and exits with code 2 (``FireExit``, a
    ``SystemExit``) before pydantic is reached.
    """

    @pytest.mark.parametrize("command", [
        ["search"],
        ["answer"],
        ["search_dataset"],
        ["answer_dataset"],
        ["evaluate", "--dataset_path", DOCS_ANSWERED],
        ["search", "hola", "--unknown_flag", "5"],
        ["not_a_command"],
    ])
    def test_usage_error(self, command: list[str]) -> None:
        """Fire exits with a usage error."""
        with pytest.raises(SystemExit):
            fire.Fire(RagCLI, command=command)
