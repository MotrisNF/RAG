
from tqdm import tqdm

from .exceptions import DatasetEvaluationError, ResultsEvaluationError
from .models import (AnsweredQuestion, MinimalSource, StudentSearchResults,
                     RagDataset, EvaluationInform)


class Evaluator:
    MAX_CNTX_LEN: int = 2000
    MIN_IOU: float = 0.05
    CUTOFFS: tuple[int, ...] = (1, 3, 5, 10)

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

    def _is_found(self,
                  correct: MinimalSource,
                  results: list[MinimalSource]
                  ) -> bool:
        return any(
            source.file_path == correct.file_path
            and self._iou(source, correct) >= self.MIN_IOU
            for source in results
        )

    def _question_recall(self,
                         correct: list[MinimalSource],
                         result: list[MinimalSource]
                         ) -> float:
        found = sum(
            1
            for source in correct
            if self._is_found(source, result)
        )

        return found / len(correct)

    def _validate_dataset(self,
                          dataset: RagDataset
                          ) -> dict[str, list[MinimalSource]]:
        if not dataset.rag_questions:
            raise DatasetEvaluationError("the dataset has no questions")

        expected: dict[str, list[MinimalSource]] = {}

        for q in dataset.rag_questions:
            if (not isinstance(q, AnsweredQuestion)
                    or not q.sources):
                raise DatasetEvaluationError(
                    f"question '{q.question_id}' has no sources: use a "
                    "valid AnsweredQuestions dataset"
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
                raise ResultsEvaluationError(
                    f"question '{question}' appears more than once"
                )
            if len(entry.retrieved_sources) > results.k:
                raise ResultsEvaluationError(
                    f"question '{question}' has "
                    f"{len(entry.retrieved_sources)} sources, but k is "
                    f"{results.k}"
                )
            for source in entry.retrieved_sources:
                start = source.first_character_index
                end = source.last_character_index
                where = f"{source.file_path} [{start}:{end}]"
                if start < 0 or end <= start:
                    raise ResultsEvaluationError(
                        f"question '{question}': source {where} has an "
                        "impossible character range"
                    )
                if end - start > self.MAX_CNTX_LEN:
                    raise ResultsEvaluationError(
                        f"question '{question}': source {where} has "
                        f"{end - start} characters (max {self.MAX_CNTX_LEN})"
                    )
            retrieved[question] = entry.retrieved_sources

        return retrieved

    def evaluate(self,
                 results: StudentSearchResults,
                 dataset: RagDataset,
                 k: int = 10
                 ) -> EvaluationInform:

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
        cutoffs = [cutoff for cutoff in self.CUTOFFS
                   if cutoff <= k and cutoff <= results.k]
        totals: dict[int, float] = {cutoff: 0.0 for cutoff in cutoffs}

        for question_id in tqdm(expected, desc="Evaluating",
                                unit="question"):
            if question_id not in retrieved:
                continue
            for cutoff in cutoffs:
                totals[cutoff] += self._question_recall(
                    expected[question_id], retrieved[question_id][:cutoff]
                )

        recall = {cutoff: total / len(expected)
                  for cutoff, total in totals.items()}

        return EvaluationInform(
           recall, missing, unknown
           )
