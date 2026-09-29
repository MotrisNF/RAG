class RagCLI:
    def index(self, max_chunk_size: int = 2000) -> None:
        ...

    def search(
            self,
            query: str,
            k: int = 10
            ) -> None:
        ...

    def search_dataset(
            self,
            dataset_path: str,
            k: int = 10,
            save_directory: str = "data/output/search_results"
            ) -> None:
        ...

    def answer(
            self,
            query: str,
            k: int = 10
            ) -> None:
        ...

    def answer_dataset(
            self,
            student_search_result_path: str,
            save_directory: str
            ) -> None:
        ...

    def evaluate(
            self,
            student_search_result_path: str,
            dataset_path: str,
            k: int = 10
            ) -> None:
        ...
