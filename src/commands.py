"""Command-line interface of the RAG system, exposed through Python Fire.

Every public method of ``RagCLI`` is a command:
``uv run python -m src <command> [options]``.
"""
import inspect
from collections import Counter
from pathlib import PurePosixPath
from typing import Annotated, Any, Callable, TypeVar, get_args, get_origin

from pydantic import ConfigDict, Field, FilePath, validate_call

from .reader import CorpusReader

F = TypeVar("F", bound=Callable[..., Any])


class CliCommand:
    """Decorator that turns a method into a validated CLI command."""

    CONFIG = ConfigDict(
        coerce_numbers_to_str=True,
        str_strip_whitespace=True
    )

    @staticmethod
    def validated(func: F) -> F:
        """Validate the arguments with pydantic and show plain types to Fire.

        1. ``validate_call`` checks every argument against its annotation
           (including ``Annotated`` constraints) and raises
           ``ValidationError`` when one is invalid.
        2. Fire reads the parameter types from ``__signature__``. A copy of
           the signature without the ``Annotated`` wrapper is stored there,
           so ``--help`` shows ``int`` instead of ``Annotated``. pydantic
           keeps using the original annotations.
        3. ``validate_call`` adds a ``raw_function`` attribute that Fire
           would list as a sub-command; it is removed.

        Args:
            func: The method to decorate.

        Returns:
            The validating wrapper, with the same type as ``func``.
        """
        wrapper = validate_call(config=CliCommand.CONFIG)(func)
        signature = inspect.signature(func)
        params = [
            param.replace(annotation=get_args(param.annotation)[0])
            if get_origin(param.annotation) is Annotated else param
            for param in signature.parameters.values()
        ]
        setattr(wrapper, "__signature__",
                signature.replace(parameters=params))
        delattr(wrapper, "raw_function")
        return wrapper


class RagCLI:
    """RAG over the vLLM codebase: index, search, answer and evaluate.

    Arguments are validated by pydantic (``CliCommand.validated``) before
    each command runs, so invalid values raise ``ValidationError``.
    """

    @CliCommand.validated
    def index(
            self,
            max_chunk_size: Annotated[int, Field(gt=0)] = 2000,
            raw_dir: Annotated[str, Field(min_length=1)] = "data/raw"
            ) -> None:
        """Chunk the corpus in data/raw/ and save the index in data/processed/.

        Args:
            max_chunk_size: Maximum length of a chunk, in characters.
            raw_dir: Folder with the corpus to index. Relative paths are
                resolved from the working directory.
        """
        result = CorpusReader(raw_dir).read()
        characters = sum(len(doc.text) for doc in result.documents)
        extensions = Counter(
            PurePosixPath(doc.file_path).suffix for doc in result.documents
        )
        print(f"Read {len(result.documents)} of {result.found} files "
              f"({characters:,} characters) from {raw_dir}")
        print("  " + " | ".join(
            f"{ext}: {count}" for ext, count in sorted(extensions.items())
        ))
        if result.skipped:
            print(f"Skipped {len(result.skipped)} unreadable files:")
            for path in result.skipped:
                print(f"  {path}")
        print(f"max_chunk_size = {max_chunk_size} "
              "(chunking not implemented yet)")

    @CliCommand.validated
    def search(
            self,
            query: Annotated[str, Field(min_length=1)],
            k: Annotated[int, Field(gt=0)] = 10
            ) -> None:
        """Print the top-k sources for a single question.

        Args:
            query: The question to search for.
            k: Number of sources to return.
        """
        raise NotImplementedError("search is not implemented yet")

    @CliCommand.validated
    def search_dataset(
            self,
            dataset_path: FilePath,
            k: Annotated[int, Field(gt=0)] = 10,
            save_directory: Annotated[
                str,
                Field(min_length=1)
                ] = "data/output/search_results"
            ) -> None:
        """Search every question of a dataset and save the results as JSON.

        Args:
            dataset_path: JSON file with the questions (RagDataset).
            k: Number of sources to return per question.
            save_directory: Folder where the StudentSearchResults file is
                written, with the same file name as the dataset.
        """
        raise NotImplementedError("search_dataset is not implemented yet")

    @CliCommand.validated
    def answer(
            self,
            query: Annotated[str, Field(min_length=1)],
            k: Annotated[int, Field(gt=0)] = 10
            ) -> None:
        """Answer a single question using the retrieved sources as context.

        Args:
            query: The question to answer.
            k: Number of sources to retrieve as context.
        """
        raise NotImplementedError("answer is not implemented yet")

    @CliCommand.validated
    def answer_dataset(
            self,
            student_search_results_path: FilePath,
            save_directory: Annotated[
                str,
                Field(min_length=1)
                ] = "data/output/search_results_and_answer"
            ) -> None:
        """Generate answers for the output of search_dataset.

        Args:
            student_search_results_path: JSON file written by search_dataset
                (StudentSearchResults).
            save_directory: Folder where the StudentSearchResultsAndAnswer
                file is written, with the same file name as the input.
        """
        raise NotImplementedError("answer_dataset is not implemented yet")

    @CliCommand.validated
    def evaluate(
            self,
            student_search_results_path: FilePath,
            dataset_path: FilePath,
            k: Annotated[int, Field(gt=0)] = 10
            ) -> None:
        """Print recall@k of search results against a ground-truth dataset.

        Args:
            student_search_results_path: JSON file written by search_dataset.
            dataset_path: JSON file with the correct sources
                (AnsweredQuestions).
            k: Largest k for which recall is reported.
        """
        raise NotImplementedError("evaluate is not implemented yet")
