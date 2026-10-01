"""Command-line interface of the RAG system, exposed through Python Fire.

Every public method of ``RagCLI`` is a command:
``uv run python -m src <command> [options]``.
"""
import functools
import inspect
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import (Annotated, Any, Callable, TypeVar, cast, get_args,
                    get_origin)

from pydantic import (BaseModel, ConfigDict, Field, FilePath, ValidationError,
                      validate_call)
from tqdm import tqdm

from .bm25_index import BM25Index
from .chunker import FixedSizeChunker
from .exceptions import (ArgError, DatasetError, DatasetEvaluationError,
                         PathError, ResultsEvaluationError)
from .evaluator import Evaluator
from .models import (MinimalSearchResults, MinimalSource, StudentSearchResults,
                     RagDataset, UnansweredQuestion, EvaluationInform)
from .reader import CorpusReader

F = TypeVar("F", bound=Callable[..., Any])
M = TypeVar("M", bound=BaseModel)


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
           (including ``Annotated`` constraints).
        2. If an ARGUMENT is invalid, its ``ValidationError`` is turned into
           an ``ArgError`` that names each option (``--k '0': ...``).
           Validation errors raised inside the command body (e.g. a broken
           dataset) are not arguments and are left untouched: they are
           told apart by their title, which is the method name only for
           argument errors.
        3. Fire reads the parameter types from ``__signature__``. A copy of
           the signature without the ``Annotated`` wrapper is stored there,
           so ``--help`` shows ``int`` instead of ``Annotated``. pydantic
           keeps using the original annotations.

        Args:
            func: The method to decorate.

        Returns:
            The validating command, with the same type as ``func``.
        """
        validator = validate_call(config=CliCommand.CONFIG)(func)
        signature = inspect.signature(func)
        names = list(signature.parameters)

        @functools.wraps(func)
        def command(*args: Any, **kwargs: Any) -> Any:
            """Run ``func`` after validating its arguments."""
            try:
                return validator(*args, **kwargs)
            except ValidationError as e:
                if e.title != func.__qualname__:
                    raise
                raise ArgError(e, names) from e

        params = [
            param.replace(annotation=get_args(param.annotation)[0])
            if get_origin(param.annotation) is Annotated else param
            for param in signature.parameters.values()
        ]
        setattr(command, "__signature__",
                signature.replace(parameters=params))
        return cast(F, command)


class RagCLI:
    """RAG over the vLLM codebase: index, search, answer and evaluate.

    Arguments are validated by pydantic (``CliCommand.validated``) before
    each command runs, so invalid values raise ``ValidationError``.

    Attributes:
        MAX_IDS_SHOWN: How many question ids ``evaluate`` lists before
            summarising the rest.
        SEPARATOR: Line between the sections of the ``evaluate`` report.
    """

    MAX_IDS_SHOWN: int = 5
    SEPARATOR: str = "=" * 35

    @CliCommand.validated
    def index(
            self,
            max_chunk_size: Annotated[int, Field(gt=0)] = 2000,
            raw_dir: Annotated[str, Field(min_length=1)] = "data/raw",
            index_dir: Annotated[str, Field(min_length=1)] = "data/processed"
            ) -> None:
        """Read the corpus in raw_dir, build the search index, save it.

        Steps: read the files (CorpusReader), split them into chunks
        (FixedSizeChunker), build the BM25 index and save it in index_dir,
        where the search commands load it from.

        Args:
            max_chunk_size: Maximum length of a chunk, in characters (the
                moulinette rejects sources longer than 2000).
            raw_dir: Folder with the corpus to index. Relative paths are
                resolved from the working directory.
            index_dir: Folder where the index is saved; created if needed.
                A previous index there is replaced.
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

        chunker = FixedSizeChunker(max_chunk_size)
        chunks = chunker.chunk_documents(result.documents)

        print(f"{len(chunks)} chunks (max {max_chunk_size} characters)")
        bm25 = BM25Index.build(chunks)
        bm25.save(Path(index_dir))
        print(f"Ingestion complete! Index saved under {index_dir}/")

    @CliCommand.validated
    def search(
            self,
            query: Annotated[str, Field(min_length=1)],
            k: Annotated[int, Field(gt=0)] = 10,
            index_dir: Annotated[str, Field(min_length=1)] = "data/processed"
            ) -> None:
        """Print the top-k sources for a single question, as JSON.

        The output is a StudentSearchResults with one entry, the same
        format search_dataset writes, so both commands behave alike.

        Args:
            query: The question to search for.
            k: Number of sources to return.
            index_dir: Folder where the index command saved the index.
        """
        bm25 = BM25Index.load(Path(index_dir))
        question = UnansweredQuestion(question=query)
        results = StudentSearchResults(
            search_results=[self._search_question(bm25, question, k)],
            k=k,
        )
        print(results.model_dump_json(indent=2))

    @staticmethod
    def _open_file_save(path: Path, model: type[M]) -> M:
        """Read a JSON file and validate it with a pydantic model.

        Shared by every command that reads a JSON input (datasets, search
        results), so the two possible failures are handled in one place.

        Args:
            path: The JSON file to read.
            model: The model the content must match (e.g. ``RagDataset``).

        Returns:
            The validated content, as an instance of ``model``.

        Raises:
            PathError: The file cannot be read, or is not UTF-8 text.
            DatasetError: The content is not valid JSON for ``model``.
        """
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            raise PathError(path, e) from e
        try:
            return model.model_validate_json(text)
        except ValidationError as e:
            raise DatasetError(str(path), e) from e

    @staticmethod
    def _search_question(bm25: BM25Index, question: UnansweredQuestion,
                         k: int) -> MinimalSearchResults:
        """Search one question and keep only the source locations.

        Each hit becomes a MinimalSource (file path and character range):
        the chunk text and the score are not part of the output format.

        Args:
            bm25: A loaded index.
            question: The question, with its id.
            k: Number of sources to retrieve.

        Returns:
            The question with its ranked sources, best first.
        """
        sources = [
            MinimalSource(
                file_path=hit.chunk.file_path,
                first_character_index=hit.chunk.first_character_index,
                last_character_index=hit.chunk.last_character_index,
            )
            for hit in bm25.search(question.question, k)
        ]
        return MinimalSearchResults(
            question_id=question.question_id,
            question=question.question,
            retrieved_sources=sources,
        )

    @CliCommand.validated
    def search_dataset(
            self,
            dataset_path: FilePath,
            k: Annotated[int, Field(gt=0)] = 10,
            save_directory: Annotated[
                str,
                Field(min_length=1)
                ] = "data/output/search_results/UnansweredQuestions",
            index_dir: Annotated[str, Field(min_length=1)] = "data/processed"
            ) -> None:
        """Search every question of a dataset and save the results as JSON.

        Args:
            dataset_path: JSON file with the questions (RagDataset).
            k: Number of sources to return per question.
            save_directory: Folder where the StudentSearchResults file is
                written, with the same file name as the dataset.
            index_dir: Folder where the index command saved the index.
        """
        bm25 = BM25Index.load(Path(index_dir))
        dataset = self._open_file_save(dataset_path, RagDataset)

        results_list: list[MinimalSearchResults] = []
        for question in tqdm(
                        dataset.rag_questions,
                        desc="Searching in the index",
                        unit="question"
                        ):
            results_list.append(self._search_question(bm25, question, k))
        output = StudentSearchResults(search_results=results_list, k=k)
        try:
            Path(save_directory).mkdir(parents=True, exist_ok=True)
            path = Path(save_directory) / dataset_path.name
            path.write_text(output.model_dump_json(indent=2),
                            encoding="utf-8")
        except OSError as e:
            raise PathError(save_directory, e) from e
        print(f"File saved in '{path}'")

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
                ] = ("data/output/search_results_and_answer/"
                     "UnansweredQuestions")
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
            k: Annotated[int, Field(gt=0)] = 10,
            show_failed: bool = False
            ) -> None:
        """Print recall@k of search results against a ground-truth dataset.

        Same metric as the moulinette's evaluate_student_search_results;
        with show_failed it also lists, like its list_valid_questions, the
        questions whose sources were not found.

        Args:
            student_search_results_path: JSON file written by search_dataset.
            dataset_path: JSON file with the correct sources
                (AnsweredQuestions).
            k: Largest k for which recall is reported.
            show_failed: List every question not found within the first k
                results, with its text.
        """

        dataset = self._open_file_save(
            dataset_path,
            RagDataset)
        retrived = self._open_file_save(
            student_search_results_path,
            StudentSearchResults)
        try:
            evaluation: EvaluationInform = Evaluator().evaluate(
                retrived, dataset, k
            )
        except DatasetEvaluationError as e:
            raise DatasetEvaluationError(
                f"dataset '{dataset_path}': {e.problem}"
            ) from e
        except ResultsEvaluationError as e:
            raise ResultsEvaluationError(
                f"results '{student_search_results_path}': {e.problem}"
            ) from e
        print("Student data is valid: True")
        print("Evaluation Results")
        print(self.SEPARATOR)
        for cutoff, value in evaluation.recall.items():
            print(f"Recall@{cutoff}: {value:.3f} ", end="")
        print()
        self._print_ids(
            f"Your results contain {len(evaluation.unknown)} questions "
            "that are not in the dataset (ignored):",
            evaluation.unknown,
        )
        self._print_ids(
            f"Your results are missing {len(evaluation.missing)} "
            "questions (counted as 0):",
            evaluation.missing,
        )
        total = len(dataset.rag_questions)
        print(self.SEPARATOR)
        print(f"Found within k={min(k, retrived.k)}: "
              f"{total - len(evaluation.failed)}/{total} questions")
        if show_failed:
            self._print_failed(evaluation.failed, dataset)
        elif evaluation.failed:
            print("Run with --show_failed to list the questions not found.")
        print(self.SEPARATOR)

    @classmethod
    def _print_failed(cls, failed: list[str], dataset: RagDataset) -> None:
        """Print every question that was not found, with its text.

        Args:
            failed: Ids of the questions not found, in dataset order.
            dataset: The dataset, to look up the text of each question.
        """
        if not failed:
            return
        texts = {q.question_id: q.question for q in dataset.rag_questions}
        print(cls.SEPARATOR)
        print(f"Questions not found ({len(failed)}):")
        for question_id in failed:
            print(f"  [FAILED] {question_id}  \"{texts[question_id]}\"")

    @classmethod
    def _print_ids(cls, title: str, ids: list[str]) -> None:
        """Print a section with some question ids, if there are any.

        At most ``MAX_IDS_SHOWN`` ids are listed; the rest are counted.

        Args:
            title: First line of the section.
            ids: Question ids to show; nothing is printed when empty.
        """
        if not ids:
            return
        print(cls.SEPARATOR)
        print(title)
        for question_id in ids[:cls.MAX_IDS_SHOWN]:
            print(f"  {question_id}")
        hidden = len(ids) - cls.MAX_IDS_SHOWN
        if hidden > 0:
            print(f"  ... and {hidden} more")
