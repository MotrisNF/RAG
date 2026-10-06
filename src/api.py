from typing import Annotated

from fastapi import FastAPI, Query


from .base_retriever import Retriever
from .commands import RagCLI
from .models import StudentSearchResults, UnansweredQuestion


class RagApi:
    def __init__(self, retriever: Retriever) -> None:
        self._retriever = retriever
        self.app = FastAPI(title="RAG against the machine",
                           description="Placeholder")
        self.app.get("/health")(self.health)
        self.app.get("/search")(self.search)

    def health(self) -> dict[str, str]:
        return {"status": "ok"}

    def search(self,
               query: Annotated[str, Query(min_length=1)],
               k: Annotated[int, Query(gt=0)] = 10
               ) -> StudentSearchResults:
        question = UnansweredQuestion(question=query)
        results = StudentSearchResults(
            search_results=[RagCLI._search_question(
                self._retriever,
                question,
                k
                )],
            k=k
        )
        self._retriever.flush()
        return results
