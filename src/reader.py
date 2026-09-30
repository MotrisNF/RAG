from pathlib import Path

from tqdm import tqdm

from .exceptions import CorpusError
from .models import CorpusReadResult, Document


class CorpusReader:
    EXTENSIONS: frozenset[str] = frozenset({".py", ".md", ".txt"})
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

    def __init__(self, path_to_read: Path | str) -> None:
        self._path_to_read = Path(path_to_read).resolve()

    def _list_files(self) -> list[Path]:
        return sorted(
            path for path in self._path_to_read.rglob("*")
            if path.suffix in self.EXTENSIONS and path.is_file()
        )

    def _relative_path(self, file: Path) -> str:
        try:
            return file.relative_to(self.PROJECT_ROOT).as_posix()
        except ValueError as e:
            raise CorpusError(
                f"{file} is outside the project root "
                f"{self.PROJECT_ROOT}"
            ) from e

    def _read_file(self, file: Path) -> Document | None:
        try:
            text = file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None
        return Document(file_path=self._relative_path(file), text=text)

    def read(self) -> CorpusReadResult:
        if not self._path_to_read.is_dir():
            raise CorpusError(
                f"Corpus folder not found: {self._path_to_read}"
            )
        if not self._path_to_read.is_relative_to(self.PROJECT_ROOT):
            raise CorpusError(
                f"Corpus folder {self._path_to_read} is outside the "
                f"project root {self.PROJECT_ROOT}"
            )
        files = self._list_files()
        if not files:
            raise CorpusError(
                "No valid files to read "
                f"in {self._path_to_read}"
            )

        files_read: list[Document] = []
        skipped: list[str] = []
        for file in tqdm(files, desc="Reading corpus", unit="file"):
            document = self._read_file(file)
            if document is not None:
                files_read.append(document)
            else:
                skipped.append(self._relative_path(file))
        if not files_read:
            raise CorpusError(
                f"No readable files in {self._path_to_read}: "
                f"{len(files)} found, none could be read as UTF-8 text"
            )
        return CorpusReadResult(documents=files_read, skipped=skipped)
