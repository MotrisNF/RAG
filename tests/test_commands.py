"""Tests for the ``CliCommand.validated`` decorator in ``src.commands``."""
import inspect
from pathlib import Path
from typing import Annotated, get_origin

import pytest
from pydantic import Field, FilePath, ValidationError

from src.commands import CliCommand, RagCLI
from src.exceptions import ArgError
from src.models import RagDataset

COMMANDS = ["index", "search", "search_dataset", "answer",
            "answer_dataset", "evaluate"]


class Sample:
    """A small class decorated like ``RagCLI``."""

    @CliCommand.validated
    def run(self, query: Annotated[str, Field(min_length=1)],
            k: Annotated[int, Field(gt=0)] = 10,
            path: FilePath | None = None) -> str:
        """Return the received values.

        Args:
            query: Text.
            k: Number.
            path: Optional file.
        """
        return f"{query}|{k}|{path}"


class TestSignature:
    """What Fire sees when it inspects a command."""

    def test_annotated_removed(self) -> None:
        """``Annotated[int, ...]`` is shown as plain ``int``."""
        params = inspect.signature(Sample.run).parameters
        assert params["query"].annotation is str
        assert params["k"].annotation is int

    def test_other_annotations_kept(self) -> None:
        """Annotations without ``Annotated`` are left as they are."""
        params = inspect.signature(Sample.run).parameters
        assert params["path"].annotation == FilePath | None

    def test_defaults_kept(self) -> None:
        """Default values survive the decoration."""
        params = inspect.signature(Sample.run).parameters
        assert params["k"].default == 10
        assert params["query"].default is inspect.Parameter.empty

    def test_name_and_docstring_kept(self) -> None:
        """Fire uses them for the help text."""
        assert Sample.run.__name__ == "run"
        assert (Sample.run.__doc__ or "").startswith("Return the received")

    def test_raw_function_removed(self) -> None:
        """``validate_call``'s helper attribute is not listed by Fire."""
        assert not hasattr(Sample.run, "raw_function")

    @pytest.mark.parametrize("command", COMMANDS)
    def test_rag_cli_signatures_clean(self, command: str) -> None:
        """No command of ``RagCLI`` exposes ``Annotated`` to Fire."""
        params = inspect.signature(getattr(RagCLI, command)).parameters
        assert all(get_origin(p.annotation) is not Annotated
                   for p in params.values())


class TestValidation:
    """pydantic still validates with the full annotations."""

    def test_valid(self) -> None:
        """Correct values reach the method."""
        assert Sample().run("hi", 3) == "hi|3|None"

    def test_constraints_enforced(self) -> None:
        """``Field`` constraints still apply."""
        with pytest.raises(ArgError):
            Sample().run("hi", 0)
        with pytest.raises(ArgError):
            Sample().run("")

    def test_numbers_to_str(self) -> None:
        """``coerce_numbers_to_str``: a number becomes text."""
        assert Sample().run(123) == "123|10|None"  # type: ignore[arg-type]

    def test_strip_whitespace(self) -> None:
        """``str_strip_whitespace``: blank text is empty, hence invalid."""
        assert Sample().run("  hi  ") == "hi|10|None"
        with pytest.raises(ArgError):
            Sample().run("   ")

    def test_str_to_int(self) -> None:
        """Numeric text is converted to ``int``."""
        assert Sample().run("hi", "7") == "hi|7|None"  # type: ignore[arg-type]

    def test_file_path(self, tmp_path: Path) -> None:
        """``FilePath`` accepts an existing file and rejects others."""
        file = tmp_path / "a.json"
        file.write_text("{}", encoding="utf-8")
        assert Sample().run("hi", path=file).endswith(str(file))
        with pytest.raises(ArgError):
            Sample().run("hi", path=tmp_path / "missing.json")
        with pytest.raises(ArgError):
            Sample().run("hi", path=tmp_path)


class TestErrors:
    """What the decorated command raises."""

    def test_names_the_option(self) -> None:
        """``ArgError`` names each invalid option and its value."""
        with pytest.raises(ArgError) as info:
            Sample().run("hi", 0)
        assert "--k '0': Input should be greater than 0" in str(info.value)

    def test_keeps_cause(self) -> None:
        """The original ``ValidationError`` is chained."""
        with pytest.raises(ArgError) as info:
            Sample().run("")
        assert isinstance(info.value.__cause__, ValidationError)

    def test_body_errors_untouched(self) -> None:
        """A ``ValidationError`` raised inside the command is not an
        argument error: it goes through unchanged.
        """
        class Broken:
            @CliCommand.validated
            def run(self, k: int = 1) -> None:
                RagDataset.model_validate({})

        with pytest.raises(ValidationError) as info:
            Broken().run(3)
        assert info.value.title == "RagDataset"


class TestConfig:
    """The shared configuration."""

    def test_config_values(self) -> None:
        """The two coercion options are enabled."""
        assert CliCommand.CONFIG.get("coerce_numbers_to_str") is True
        assert CliCommand.CONFIG.get("str_strip_whitespace") is True
