"""Exception hierarchy for the RAG project."""
from pydantic import ValidationError


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
