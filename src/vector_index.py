from typing import TYPE_CHECKING

import numpy as np

from .exceptions import RagError
from .models import Chunk, WindowChunk
from .tokenizer_bm import Tokenizer

if TYPE_CHECKING:
    # Only for mypy: importing it for real would load PyTorch at start-up.
    from sentence_transformers import SentenceTransformer


class VectorIndex:
    MAX_WINDOW_SIZE = 400
    MODEL_NAME = "minishlab/potion-base-8M"

    def __init__(self, vectors: np.ndarray, owners: np.ndarray) -> None:
        self._vectors = vectors
        self._owners = owners
        self._model: "SentenceTransformer | None" = None

    @classmethod
    def _load_model(cls) -> "SentenceTransformer":
        """Load the embedding model, from the local cache first.

        Same idea as ``Generator._from_pretrained``: with the model cached
        no request is sent to the Hugging Face Hub (a request that never
        answers would hang the command); the Hub is only used for the
        first download.

        Returns:
            The model, on CPU.

        Raises:
            RagError: The model is not cached and cannot be downloaded.
        """
        from sentence_transformers import SentenceTransformer

        try:
            return SentenceTransformer(cls.MODEL_NAME, device="cpu",
                                       local_files_only=True)
        except OSError:
            pass
        try:
            return SentenceTransformer(cls.MODEL_NAME, device="cpu")
        except OSError as e:
            raise RagError(
                f"Cannot load {cls.MODEL_NAME}: it is not in the local "
                f"cache and the download failed ({e}). Check the internet "
                "connection, or HF_HOME if the model is cached elsewhere."
            ) from e

    def _encoder(self) -> "SentenceTransformer":
        """The embedding model, loaded the first time it is needed.

        An index read with ``load`` has no model until the first query;
        later queries (every question of ``search_dataset``) reuse it.

        Returns:
            The loaded model.
        """
        if self._model is None:
            self._model = self._load_model()
        return self._model

    @classmethod
    def _get_window(cls, text: str) -> list[str]:
        window_list: list[str] = []
        window_text: str = ""
        for line in text.splitlines(keepends=True):
            if (window_text
                    and len(window_text) + len(line) > cls.MAX_WINDOW_SIZE):
                window_list.append(window_text)
                window_text = line
                continue
            window_text += line
        if window_text != "":
            window_list.append(window_text)
        return window_list

    @classmethod
    def _window_chunk(cls, chunks: list[Chunk]) -> list[WindowChunk]:
        tokenizer = Tokenizer()
        window_chunk_list: list[WindowChunk] = []
        for chunk in chunks:
            prefix = " ".join(tokenizer.tokenize(chunk.file_path)) + "\n"
            window_text = [prefix + window
                           for window in cls._get_window(chunk.text)]
            window_chunk_list.append(
                WindowChunk(
                    windows=window_text,
                    windows_count=len(window_text)
                )
            )
        return window_chunk_list

    @classmethod
    def build(cls, chunks: list[Chunk]) -> "VectorIndex":
        window_chunks = cls._window_chunk(chunks)
        counts = [item.windows_count for item in window_chunks]
        owners = np.repeat(np.arange(len(window_chunks)), counts)
        texts = [window for item in window_chunks for window in item.windows]

        model = cls._load_model()
        vectors = model.encode(
            texts,
            batch_size=512,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        index = cls(vectors, owners)
        index._model = model
        return index
