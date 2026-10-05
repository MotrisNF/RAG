"""Read the corpus files that will be indexed."""
import os
from pathlib import Path

from tqdm import tqdm

from .exceptions import CorpusError
from .models import CorpusReadResult, Document


class CorpusReader:
    """Read the text files of a corpus folder.

    Paths are stored relative to the project root (``data/raw/...``),
    the form the moulinette compares verbatim.

    Paths are normalised without following symbolic links: ``..`` is
    removed as text, so ``--raw_dir ../`` is still outside the project,
    but ``data/raw`` may be a link to a bigger disk (``goinfre`` on 42
    computers) and stays ``data/raw/...``.

    Attributes:
        EXTENSIONS: File types that are read.
        PROJECT_ROOT: Folder the stored paths are relative to.
    """

    EXTENSIONS: frozenset[str] = frozenset({".py", ".md", ".txt"})
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

    def __init__(self, path_to_read: Path | str) -> None:
        """Store the folder to read.

        Args:
            path_to_read: Corpus folder; relative paths are taken from
                the working directory.
        """
        self._path_to_read = Path(os.path.abspath(path_to_read))

    @classmethod
    def project_path(cls, path: Path | str) -> Path:
        """Absolute path of a project file, refusing anything outside.

        A relative path is taken from the project root. ``..`` and
        absolute paths are normalised first, so ``../../.ssh/id_rsa`` or
        ``/etc/passwd`` are rejected; symbolic links are not followed.
        Used for the source files listed in a search results JSON.

        Args:
            path: A path such as ``data/raw/vllm-0.10.1/README.md``.

        Returns:
            The absolute, normalised path inside the project.

        Raises:
            CorpusError: The path is outside the project root.
        """
        absolute = Path(os.path.abspath(cls.PROJECT_ROOT / path))
        if not absolute.is_relative_to(cls.PROJECT_ROOT):
            raise CorpusError(
                f"{path} is outside the project root {cls.PROJECT_ROOT}"
            )
        return absolute

    def _list_files(self) -> list[Path]:
        """Files of the folder, recursively, with a valid extension."""
        return sorted(
            path for path in self._path_to_read.rglob("*")
            if path.suffix in self.EXTENSIONS and path.is_file()
        )

    def _relative_path(self, file: Path) -> str:
        """Path of ``file`` relative to the project root, with ``/``.

        Raises:
            CorpusError: The file is outside the project root.
        """
        try:
            return file.relative_to(self.PROJECT_ROOT).as_posix()
        except ValueError as e:
            raise CorpusError(
                f"{file} is outside the project root "
                f"{self.PROJECT_ROOT}"
            ) from e

    def _read_file(self, file: Path) -> Document | None:
        """Read one file as UTF-8, or return None if it cannot be read."""
        try:
            text = file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None
        return Document(file_path=self._relative_path(file), text=text)

    def read(self) -> CorpusReadResult:
        """Read every valid file of the folder, with a progress bar.

        Returns:
            The documents read and the paths of the skipped files.

        Raises:
            CorpusError: The folder does not exist, is outside the
                project root, or has no readable file.
        """
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
