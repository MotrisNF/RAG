"""Exception hierarchy for the RAG project."""
from pydantic import ValidationError
from pydantic_core import ErrorDetails


class RagError(Exception):
    """Base class for every error raised by the RAG project.

    Catching ``RagError`` catches all of its subclasses too.
    """

    def __init__(self, msg: str = "Unknown RagError") -> None:
        """Store a human-readable message.

        Args:
            msg: Message shown to the user.
        """
        super().__init__(msg)


class DatasetError(RagError):
    """A JSON file could not be parsed or does not match its model."""

    def __init__(self, path: str, error: ValidationError) -> None:
        """Build the message as one ``field: message`` line per error.

        Args:
            path: Path of the file that failed to validate.
            error: The original pydantic error.
        """
        lines = [f"Invalid file '{path}':"]
        for err in error.errors():
            where = ".".join(str(p) for p in err["loc"]) or "<root>"
            lines.append(f"  {where}: {err['msg']}")
        super().__init__("\n".join(lines))


class ArgError(RagError):
    """A command-line argument was rejected by validation.

    One line per invalid argument: ``--option 'value': reason``.
    """

    def __init__(self, error: ValidationError, names: list[str]) -> None:
        """Build the message from a pydantic validation error.

        Args:
            error: The original pydantic error.
            names: Parameter names of the command, in order and including
                ``self``. pydantic reports each argument by its position
                (``loc``), and Fire always passes them by position, so the
                position is turned back into the option name here.
        """
        lines = ["Invalid arguments:"]
        for err in error.errors():
            option = self._option(err["loc"], names)
            lines.append(f"  {option} '{err['input']}': {self._reason(err)}")
        super().__init__("\n".join(lines))

    @staticmethod
    def _option(loc: tuple[int | str, ...], names: list[str]) -> str:
        """``--name`` of the argument at ``loc`` (position or name)."""
        where = loc[0] if loc else None
        if isinstance(where, int) and 0 <= where < len(names):
            return f"--{names[where]}"
        if isinstance(where, str):
            return f"--{where}"
        return "argument"

    @staticmethod
    def _reason(err: ErrorDetails) -> str:
        """pydantic's message, clearer for blank text.

        Text is stripped before checking its length, but pydantic reports
        the original value, so ``' '`` would read "should have at least 1
        character".
        """
        value = err["input"]
        if (err["type"] == "string_too_short" and isinstance(value, str)
                and not value.strip()):
            return "must not be empty or blank"
        return err["msg"]


class CorpusError(RagError):
    def __init__(self, msg: str = "Something failed reading the corpus."):
        super().__init__(msg)


class UsageError(RagError):
    """The command line is malformed (unknown option, too many arguments).

    ``main`` exits with code 2 for it, like Fire's own usage errors.
    """


class IndexBuildError(RagError):
    """The search index cannot be built from the given chunks.

    Raised when there is nothing to index: no chunks at all, or chunks
    without a single token (e.g. only symbols). Without it the average
    chunk length would be 0 and BM25 would divide by zero.
    """
