"""Tests for ``src.argv_guard.ArgvGuard``."""
import pytest

from src.argv_guard import ArgvGuard
from src.commands import RagCLI
from src.exceptions import RagError, UsageError


class Sample:
    """A small component: two commands and a private method."""

    def search(self, query: str, k: int = 10) -> None:
        """Two parameters, the second with a default."""

    def index(self, max_chunk_size: int = 2000, raw_dir: str = "r") -> None:
        """Only options with defaults."""

    def kinds(self, key: str = "", kind: str = "") -> None:
        """Two parameters with the same initial."""

    def flags(self, verbose: bool = False, name: str = "") -> None:
        """A boolean flag next to a normal option."""

    def _private(self) -> None:
        """Not a command."""


@pytest.fixture
def guard() -> ArgvGuard:
    """Guard for the ``Sample`` component."""
    return ArgvGuard(Sample)


class TestAccepted:
    """Every form Fire accepts is let through unchanged."""

    @pytest.mark.parametrize("argv", [
        ["search", "hola"],
        ["search", "hola", "5"],
        ["search", "hola", "--k", "5"],
        ["search", "--k", "5", "hola"],
        ["search", "--query", "hola", "--k", "5"],
        ["search", "hola", "--k=5"],
        ["search", "hola", "-k", "5"],
        ["search", "hola", "-k=5"],
        ["search", "hola", "--k", "-3"],
        ["search", "hola", "--k=-3"],
        ["search", "hola", "--k", "2.5"],
        ["search", "-5"],
        ["index"],
        ["index", "--max_chunk_size", "500"],
        ["index", "--max-chunk-size", "500"],
        ["index", "-max_chunk_size", "500"],
        ["index", "-m", "500"],
        ["index", "-r", "data/raw"],
        ["index", "500", "data/raw"],
        ["kinds", "--key", "a", "--kind", "b"],
    ])
    def test_unchanged(self, guard: ArgvGuard, argv: list[str]) -> None:
        """The command line is returned as it came."""
        assert guard.check(argv) == argv

    @pytest.mark.parametrize("argv", [
        [],
        ["not_a_command", "--whatever"],
        ["_private"],
        ["--help"],
        ["search", "--", "--trace"],
        ["search", "hola", "-", "upper"],
    ])
    def test_left_to_fire(self, guard: ArgvGuard, argv: list[str]) -> None:
        """No command, unknown commands and text after a separator are
        Fire's business (it fails before running anything).
        """
        assert guard.check(argv) == argv


class TestHelp:
    """``--help`` never runs the command."""

    @pytest.mark.parametrize("argv", [
        ["search", "--help"],
        ["search", "hola", "--help"],
        ["search", "hola", "-h"],
        ["search", "hola", "-help"],
        ["index", "--max_chunk_size", "5", "--help"],
        ["index", "--help", "--max_chunk_size", "5"],
        ["index", "--unknown", "--help"],
    ])
    def test_rewritten(self, guard: ArgvGuard, argv: list[str]) -> None:
        """It becomes ``<command> -- --help``, Fire's help-only form."""
        assert guard.check(argv) == [argv[0], "--", "--help"]

    def test_help_after_separator_untouched(self, guard: ArgvGuard) -> None:
        """``command -- --help`` is already the right form."""
        argv = ["search", "--", "--help"]
        assert guard.check(argv) == argv


class TestUnknownOptions:
    """Options that match no parameter."""

    @pytest.mark.parametrize("argv", [
        ["index", "--chunk_size", "5"],
        ["index", "--max_chunk", "5"],
        ["index", "--max_chunk_size", "5", "--raw", "x"],
        ["index", "--chunk-size=5"],
        ["search", "hola", "--kk", "5"],
        ["search", "hola", "-x", "5"],
        ["search", "hola", "--nok"],
        ["kinds", "-k", "a"],
    ])
    def test_rejected(self, guard: ArgvGuard, argv: list[str]) -> None:
        """``UsageError`` naming the option."""
        with pytest.raises(UsageError, match="Unknown option"):
            guard.check(argv)

    def test_message_lists_valid_options(self, guard: ArgvGuard) -> None:
        """The message says which options exist."""
        with pytest.raises(UsageError) as info:
            guard.check(["index", "--chunk_size", "5"])
        message = str(info.value)
        assert "'--chunk_size'" in message
        assert "'index'" in message
        assert "--max_chunk_size, --raw_dir" in message

    def test_ambiguous_initial(self, guard: ArgvGuard) -> None:
        """``-k`` is ambiguous when two parameters start with ``k``."""
        with pytest.raises(UsageError):
            guard.check(["kinds", "-k", "a"])


class TestTooManyArguments:
    """More values than the command has parameters."""

    @pytest.mark.parametrize("argv", [
        ["search", "hola", "3", "4"],
        ["search", "How", "does", "it", "work"],
        ["search", "hola", "--k", "3", "extra"],
        ["search", "--query", "a", "--k", "3", "extra"],
        ["index", "1", "2", "3"],
    ])
    def test_rejected(self, guard: ArgvGuard, argv: list[str]) -> None:
        """``UsageError`` saying how many it takes."""
        with pytest.raises(UsageError, match="Too many arguments"):
            guard.check(argv)

    def test_message(self, guard: ArgvGuard) -> None:
        """The message gives the limit and the options."""
        with pytest.raises(UsageError) as info:
            guard.check(["search", "a", "b", "c"])
        assert "at most 2 (--query, --k)" in str(info.value)


class TestMissingValue:
    """An option written without its value."""

    @pytest.mark.parametrize("argv", [
        ["index", "--raw_dir"],
        ["index", "--max_chunk_size", "10", "--raw_dir"],
        ["index", "--raw_dir", "--max_chunk_size", "10"],
        ["index", "-m"],
        ["index", "--max-chunk-size"],
        ["search", "hola", "--k"],
        ["index", "--raw_dir", "--", "--trace"],
        ["flags", "--name"],
    ])
    def test_rejected(self, guard: ArgvGuard, argv: list[str]) -> None:
        """``UsageError`` instead of Fire turning it into ``True``."""
        with pytest.raises(UsageError, match="needs a value"):
            guard.check(argv)

    def test_message(self, guard: ArgvGuard) -> None:
        """The message names the option as written and the command."""
        with pytest.raises(UsageError) as info:
            guard.check(["index", "-m"])
        assert str(info.value) == "Option '-m' of 'index' needs a value"

    @pytest.mark.parametrize("argv", [
        ["flags", "--verbose"],
        ["flags", "--verbose", "--name", "x"],
        ["flags", "--name", "x", "--verbose"],
        ["flags", "--verbose", "False"],
        ["flags", "-v"],
    ])
    def test_bool_flag_alone_ok(self, guard: ArgvGuard,
                                argv: list[str]) -> None:
        """A ``bool`` parameter may appear without a value."""
        assert guard.check(argv) == argv

    @pytest.mark.parametrize("argv", [
        ["index", "--raw_dir="],
        ["search", "hola", "--k", "-3"],
        ["search", "hola", "--k=-3"],
    ])
    def test_value_present(self, guard: ArgvGuard, argv: list[str]) -> None:
        """An empty ``=`` value or a negative number is a value (pydantic
        judges it later).
        """
        assert guard.check(argv) == argv

    def test_help_wins(self, guard: ArgvGuard) -> None:
        """Asking for help is honoured even if a value is missing."""
        argv = ["index", "--raw_dir", "--help"]
        assert guard.check(argv) == ["index", "--", "--help"]


class TestErrorType:
    """How ``main`` handles it."""

    def test_is_rag_error(self, guard: ArgvGuard) -> None:
        """``UsageError`` is a ``RagError``."""
        with pytest.raises(RagError):
            guard.check(["index", "--nope", "1"])


class TestRagCLI:
    """The guard against the real CLI."""

    def test_commands(self) -> None:
        """The six subject commands are detected."""
        assert ArgvGuard(RagCLI)._commands() == {
            "index", "search", "search_dataset", "answer",
            "answer_dataset", "evaluate",
        }

    def test_uses_clean_signatures(self) -> None:
        """Parameters come from the decorated methods."""
        params = ArgvGuard(RagCLI)._params("index")
        assert params == ["max_chunk_size", "raw_dir", "index_dir"]

    @pytest.mark.parametrize("argv", [
        ["index", "--chunk_size", "5"],
        ["search_dataset", "--dataset", "x.json"],
        ["answer_dataset", "--student_search_result_path", "x.json"],
        ["evaluate", "--datase_path", "x.json"],
    ])
    def test_typos_rejected(self, argv: list[str]) -> None:
        """Typical typos in the subject's option names are caught."""
        with pytest.raises(UsageError):
            ArgvGuard(RagCLI).check(argv)
