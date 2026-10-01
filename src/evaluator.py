from pathlib import Path
from typing import Annotated, Any

from pydantic import FilePath, Field
from tqdm import tqdm

from .exceptions import (DatasetError, IndexNotFoundError,
                         PathError, EvaluationError, RagError)
from .models import (AnsweredQuestion, MinimalSource, StudentSearchResults,
                     RagDataset, UnansweredQuestion)


class Evaluator:
    MAX_CNTX_LEN: int = 2000

    def _iou(self,
             a: MinimalSource,
             b: MinimalSource
             ) -> float:
        intersection = max(0, min(a.last_character_index,
                           b.last_character_index)
                           - max(a.first_character_index,
                                 b.first_character_index))
        union = ((a.last_character_index - a.first_character_index)
                 + (b.last_character_index - b.first_character_index)
                 - intersection)
        return intersection / union

    def _validate_dataset(self,
                          dataset: RagDataset
                          ) -> dict[str, list[MinimalSource]]:
        if not dataset.rag_questions:
            raise EvaluationError("the dataset has no questions")

        expected: dict[str, list[MinimalSource]] = {}

        for q in dataset.rag_questions:
            if not isinstance(q, AnsweredQuestion):
                raise EvaluationError(
                    f"question '{q.question_id}' has no sources: use an "
                    "AnsweredQuestions dataset"
                )
            expected[q.question_id] = q.sources

        return expected

    def _validate_results(self,
                          results: StudentSearchResults
                          ) -> dict[str, list[MinimalSource]]:
        retrieved: dict[str, list[MinimalSource]] = {}

        for entry in results.search_results:
            question = entry.question_id
            if question in retrieved:
                raise EvaluationError(
                    f"question '{question}' appears more than once"
                )
            if len(entry.retrieved_sources) > results.k:
                raise EvaluationError(
                    f"question '{question}' has "
                    f"{len(entry.retrieved_sources)} sources, but k is "
                    f"{results.k}"
                )
            for source in entry.retrieved_sources:
                start = source.first_character_index
                end = source.last_character_index
                where = f"{source.file_path} [{start}:{end}]"
                if start < 0 or end <= start:
                    raise EvaluationError(
                        f"question '{question}': source {where} has an "
                        "impossible character range"
                    )
                if end - start > self.MAX_CNTX_LEN:
                    raise EvaluationError(
                        f"question '{question}': source {where} has "
                        f"{end - start} characters (max {self.MAX_CNTX_LEN})"
                    )
            retrieved[question] = entry.retrieved_sources

        return retrieved

    def evaluate(self,
                 results: StudentSearchResults,
                 dataset: RagDataset,
                 k: int = 10
                 ) -> None:

        expected = self._validate_dataset(dataset)
        retrieved = self._validate_results(results)

        missing: list[str] = [
            question_id
            for question_id in expected
            if question_id not in retrieved
            ]
        unknown: list[str] = [
            question_id
            for question_id in retrieved
            if question_id not in expected
            ]
        
        raise NotImplementedError("It's not done yet")
