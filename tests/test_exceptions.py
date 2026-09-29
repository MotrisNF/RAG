"""Tests for the exception hierarchy in ``src.exceptions``."""
import fire
import pytest
from pydantic import ValidationError

from src.commands import RagCLI
from src.exceptions import ArgError, DatasetError, RagError
from src.models import RagDataset, UnansweredQuestion


class ErrorFactory:
    """Build real pydantic ``ValidationError`` objects for the tests."""

    @staticmethod
    def from_model(data: dict[str, object]) -> ValidationError:
        """Validate ``data`` as an UnansweredQuestion and return the error."""
        with pytest.raises(ValidationError) as info:
            UnansweredQuestion.model_validate(data)
        return info.value

    @staticmethod
    def from_json(text: str) -> ValidationError:
        """Validate ``text`` as a RagDataset and return the error."""
        with pytest.raises(ValidationError) as info:
            RagDataset.model_validate_json(text)
        return info.value

    @staticmethod
    def from_cli(command: list[str]) -> ValidationError:
        """Run ``command`` through Fire and return the validation error."""
        with pytest.raises(ValidationError) as info:
            fire.Fire(RagCLI, command=command)
        return info.value


class TestRagError:
    """The base class of the hierarchy."""

    def test_default_message(self) -> None:
        """Without arguments it uses the default message."""
        assert str(RagError()) == "Unknown RagError"

    def test_custom_message(self) -> None:
        """The message given is the one shown."""
        assert str(RagError("Index not built")) == "Index not built"

    def test_is_exception(self) -> None:
        """It can be caught as any other exception."""
        with pytest.raises(Exception):
            raise RagError("boom")


class TestDatasetError:
    """Formatting of dataset validation errors."""

    def test_header_has_path(self) -> None:
        """The first line names the file."""
        error = ErrorFactory.from_model({})
        lines = str(DatasetError("x.json", error)).splitlines()
        assert lines[0] == "Invalid file 'x.json':"

    def test_one_line_per_error(self) -> None:
        """Every pydantic error gets its own ``field: message`` line."""
        error = ErrorFactory.from_model({"question_id": 5})
        lines = str(DatasetError("x.json", error)).splitlines()
        assert len(lines) == 1 + error.error_count()
        assert "  question_id: Input should be a valid string" in lines
        assert "  question: Field required" in lines

    def test_nested_location(self) -> None:
        """Nested fields are joined with dots, list positions included."""
        error = ErrorFactory.from_json('{"rag_questions": [{}]}')
        message = str(DatasetError("x.json", error))
        assert "rag_questions.0" in message

    def test_invalid_json_uses_root(self) -> None:
        """Broken JSON has no field, so ``<root>`` is shown."""
        error = ErrorFactory.from_json('{"rag_questions": [,]}')
        message = str(DatasetError("x.json", error))
        assert "  <root>: Invalid JSON" in message

    def test_caught_as_rag_error(self) -> None:
        """``except RagError`` catches it."""
        error = ErrorFactory.from_model({})
        with pytest.raises(RagError):
            raise DatasetError("x.json", error) from error


class TestArgError:
    """Formatting of command-line argument errors."""

    def test_header(self) -> None:
        """The first line says the arguments are invalid."""
        error = ErrorFactory.from_cli(["search", "hola", "--k", "0"])
        lines = str(ArgError(error)).splitlines()
        assert lines[0] == "Invalid arguments:"

    def test_shows_value_and_reason(self) -> None:
        """Each line shows the received value and why it was rejected."""
        error = ErrorFactory.from_cli(["search", "hola", "--k", "0"])
        assert "  '0': Input should be greater than 0" in str(ArgError(error))

    def test_missing_file(self) -> None:
        """A missing input file is reported with its path."""
        error = ErrorFactory.from_cli(
            ["search_dataset", "--dataset_path", "nope.json"])
        assert "'nope.json'" in str(ArgError(error))

    def test_several_errors(self) -> None:
        """All invalid arguments are listed, one per line."""
        error = ErrorFactory.from_cli(["search", "", "--k", "abc"])
        lines = str(ArgError(error)).splitlines()
        assert len(lines) == 1 + error.error_count()
        assert error.error_count() == 2

    def test_caught_as_rag_error(self) -> None:
        """``except RagError`` catches it."""
        error = ErrorFactory.from_cli(["search", "hola", "--k", "0"])
        with pytest.raises(RagError):
            raise ArgError(error) from error
