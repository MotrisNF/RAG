"""Tests for ``src.reader.CorpusReader``.

Most tests build a small fake project in ``tmp_path`` and point
``CorpusReader.PROJECT_ROOT`` at it, so relative paths are computed
exactly as in the real project (``data/raw/...``).
"""
import json
from pathlib import Path

import pytest

from src.exceptions import CorpusError, RagError
from src.models import CorpusReadResult, Document
from src.reader import CorpusReader

REAL_RAW = CorpusReader.PROJECT_ROOT / "data" / "raw"
DATASETS = CorpusReader.PROJECT_ROOT / "data" / "datasets"
NOT_UTF8 = b"\xff\xfe\x00 not utf-8 \x80"


class FakeProject:
    """A throw-away project tree inside ``tmp_path``."""

    def __init__(self, root: Path) -> None:
        """Create ``data/raw`` under ``root``.

        Args:
            root: Folder acting as the project root.
        """
        self.root = root
        self.raw = root / "data" / "raw"
        self.raw.mkdir(parents=True)

    def write(self, relative: str, content: str | bytes = "x") -> Path:
        """Write a file under ``data/raw`` (parent folders included).

        Args:
            relative: Path relative to ``data/raw``.
            content: Text (UTF-8) or raw bytes.

        Returns:
            The absolute path of the written file.
        """
        path = self.raw / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
        return path

    def read(self) -> CorpusReadResult:
        """Read ``data/raw`` with a ``CorpusReader``."""
        return CorpusReader(self.raw).read()


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeProject:
    """Fake project whose root is ``tmp_path``."""
    monkeypatch.setattr(CorpusReader, "PROJECT_ROOT", tmp_path)
    return FakeProject(tmp_path)


class TestClassAttributes:
    """Fixed configuration of the reader."""

    def test_extensions(self) -> None:
        """Python, Markdown and text files are read."""
        assert CorpusReader.EXTENSIONS == {".py", ".md", ".txt"}

    def test_extensions_are_immutable(self) -> None:
        """The shared class attribute cannot be modified by accident."""
        assert isinstance(CorpusReader.EXTENSIONS, frozenset)

    def test_project_root(self) -> None:
        """The root is the folder that contains ``src``."""
        root = CorpusReader.PROJECT_ROOT
        assert root.is_absolute()
        assert (root / "src" / "reader.py").is_file()
        assert (root / "pyproject.toml").is_file()


class TestFileSelection:
    """Which files end up as documents."""

    def test_only_allowed_extensions(self, project: FakeProject) -> None:
        """Other extensions are ignored."""
        for name in ("a.py", "b.md", "c.txt", "d.png", "e.json", "f.cpp",
                     "g", "h.pyc"):
            project.write(name)
        paths = [d.file_path for d in project.read().documents]
        assert paths == ["data/raw/a.py", "data/raw/b.md", "data/raw/c.txt"]

    def test_recursive(self, project: FakeProject) -> None:
        """Files in nested sub-folders are found."""
        project.write("top.md")
        project.write("a/b/c/deep.py")
        paths = {d.file_path for d in project.read().documents}
        assert paths == {"data/raw/top.md", "data/raw/a/b/c/deep.py"}

    def test_hidden_folders_included(self, project: FakeProject) -> None:
        """Folders starting with a dot (like ``.buildkite``) are read."""
        project.write(".hidden/script.py")
        paths = [d.file_path for d in project.read().documents]
        assert paths == ["data/raw/.hidden/script.py"]

    def test_directory_with_extension_ignored(
            self, project: FakeProject) -> None:
        """A folder named ``*.md`` is not a file."""
        (project.raw / "folder.md").mkdir()
        project.write("real.md")
        paths = [d.file_path for d in project.read().documents]
        assert paths == ["data/raw/real.md"]

    def test_sorted_and_deterministic(self, project: FakeProject) -> None:
        """Documents come in path order, the same on every call."""
        for name in ("z.md", "a.md", "m/b.py", "b.txt"):
            project.write(name)
        first = [d.file_path for d in project.read().documents]
        second = [d.file_path for d in project.read().documents]
        assert first == sorted(first)
        assert first == second


class TestDocumentContent:
    """What each document contains."""

    def test_text_is_identical(self, project: FakeProject) -> None:
        """The text is exactly the file content."""
        content = "# Title\n\nSome text.\n\n```py\nx = 1\n```\n"
        project.write("doc.md", content)
        assert project.read().documents[0].text == content

    def test_unicode(self, project: FakeProject) -> None:
        """Non-ASCII UTF-8 text is read correctly."""
        content = "Café, ñandú, 日本語, emoji 🚀\n"
        project.write("u.md", content)
        assert project.read().documents[0].text == content

    def test_empty_file_is_read(self, project: FakeProject) -> None:
        """An empty file is a valid document with empty text."""
        project.write("empty.py", "")
        project.write("other.py", "x = 1\n")
        docs = project.read().documents
        assert Document(file_path="data/raw/empty.py", text="") in docs

    def test_offsets_match_file(self, project: FakeProject) -> None:
        """Slicing the text gives the same characters as the file."""
        content = "line one\nline two\nline three\n"
        path = project.write("x.txt", content)
        text = project.read().documents[0].text
        assert text[9:17] == path.read_text(encoding="utf-8")[9:17]
        assert text[9:17] == "line two"

    def test_returns_documents(self, project: FakeProject) -> None:
        """The result holds ``Document`` objects."""
        project.write("a.md")
        result = project.read()
        assert isinstance(result, CorpusReadResult)
        assert all(isinstance(d, Document) for d in result.documents)


class TestPaths:
    """The ``file_path`` stored in each document."""

    def test_relative_to_project_root(self, project: FakeProject) -> None:
        """Paths start at ``data/raw``, never absolute."""
        project.write("vllm-0.10.1/docs/features/lora.md")
        path = project.read().documents[0].file_path
        assert path == "data/raw/vllm-0.10.1/docs/features/lora.md"
        assert not Path(path).is_absolute()

    def test_forward_slashes(self, project: FakeProject) -> None:
        """Paths always use ``/``."""
        project.write("a/b/c.md")
        path = project.read().documents[0].file_path
        assert "\\" not in path
        assert path.count("/") == 4

    def test_accepts_str_and_path(self, project: FakeProject) -> None:
        """The folder can be given as ``str`` or ``Path``."""
        project.write("a.md")
        from_path = CorpusReader(project.raw).read()
        from_str = CorpusReader(str(project.raw)).read()
        assert from_path == from_str

    def test_relative_input_folder(
            self, project: FakeProject,
            monkeypatch: pytest.MonkeyPatch) -> None:
        """A relative folder is resolved from the working directory."""
        project.write("a.md")
        monkeypatch.chdir(project.root)
        result = CorpusReader("data/raw").read()
        paths = [d.file_path for d in result.documents]
        assert paths == ["data/raw/a.md"]

    def test_same_paths_from_other_working_dir(
            self, project: FakeProject,
            monkeypatch: pytest.MonkeyPatch) -> None:
        """Stored paths do not depend on where the program is launched."""
        project.write("a.md")
        monkeypatch.chdir(project.raw)
        paths = [d.file_path for d in CorpusReader(".").read().documents]
        assert paths == ["data/raw/a.md"]


class TestSkippedFiles:
    """Files that exist but cannot be read."""

    def test_not_utf8_is_skipped(self, project: FakeProject) -> None:
        """A file that is not valid UTF-8 goes to ``skipped``."""
        project.write("good.md", "ok")
        project.write("bad.md", NOT_UTF8)
        result = project.read()
        assert [d.file_path for d in result.documents] == ["data/raw/good.md"]
        assert result.skipped == ["data/raw/bad.md"]

    def test_found_counts_both(self, project: FakeProject) -> None:
        """``found`` = read + skipped."""
        project.write("a.md")
        project.write("b.md")
        project.write("c.md", NOT_UTF8)
        result = project.read()
        assert len(result.documents) == 2
        assert len(result.skipped) == 1
        assert result.found == 3

    def test_nothing_skipped(self, project: FakeProject) -> None:
        """When every file is readable ``skipped`` is empty."""
        project.write("a.md")
        assert project.read().skipped == []

    def test_unreadable_file_skipped(
            self, project: FakeProject,
            monkeypatch: pytest.MonkeyPatch) -> None:
        """An ``OSError`` while reading (e.g. permissions) skips the file."""
        project.write("good.md")
        bad = project.write("locked.md")
        original = Path.read_text

        def fake_read_text(self: Path, *args: object,
                           **kwargs: object) -> str:
            """Fail with PermissionError for the locked file only."""
            if self == bad:
                raise PermissionError("denied")
            return original(self, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(Path, "read_text", fake_read_text)
        result = project.read()
        assert result.skipped == ["data/raw/locked.md"]
        assert len(result.documents) == 1


class TestErrors:
    """Every failure is a ``CorpusError`` with a clear message."""

    def test_folder_not_found(self, project: FakeProject) -> None:
        """A missing folder is reported as not found."""
        with pytest.raises(CorpusError, match="Corpus folder not found"):
            CorpusReader(project.root / "nope").read()

    def test_path_is_a_file(self, project: FakeProject) -> None:
        """A file instead of a folder is reported as not found."""
        path = project.write("a.md")
        with pytest.raises(CorpusError, match="Corpus folder not found"):
            CorpusReader(path).read()

    def test_empty_folder(self, project: FakeProject) -> None:
        """A folder without files has nothing to read."""
        with pytest.raises(CorpusError, match="No valid files to read in"):
            project.read()

    def test_no_allowed_extension(self, project: FakeProject) -> None:
        """Files exist, but none with an allowed extension."""
        project.write("image.png")
        project.write("data.json")
        with pytest.raises(CorpusError, match="No valid files to read in"):
            project.read()

    def test_nothing_readable(self, project: FakeProject) -> None:
        """Every candidate file fails to read."""
        project.write("a.md", NOT_UTF8)
        project.write("b.py", NOT_UTF8)
        with pytest.raises(CorpusError, match="No readable files") as info:
            project.read()
        assert "2 found" in str(info.value)

    def test_is_rag_error(self, project: FakeProject) -> None:
        """``main`` catches it through ``RagError``."""
        with pytest.raises(RagError):
            CorpusReader(project.root / "nope").read()

    def test_constructor_does_not_touch_disk(self, tmp_path: Path) -> None:
        """Errors appear on ``read()``, not when creating the reader."""
        CorpusReader(tmp_path / "does" / "not" / "exist")


class TestProjectBoundary:
    """The corpus folder must be inside the project root.

    Layout used here::

        tmp_path/                 <- parent of the project
        ├── project/              <- PROJECT_ROOT
        │   └── data/raw/a.md
        └── elsewhere/b.md        <- sibling, outside the project
    """

    @pytest.fixture
    def root(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """Build the layout and point ``PROJECT_ROOT`` at ``project``."""
        root = tmp_path / "project"
        (root / "data" / "raw").mkdir(parents=True)
        (root / "data" / "raw" / "a.md").write_text("a", encoding="utf-8")
        (tmp_path / "elsewhere").mkdir()
        (tmp_path / "elsewhere" / "b.md").write_text("b", encoding="utf-8")
        monkeypatch.setattr(CorpusReader, "PROJECT_ROOT", root)
        return root

    def test_sibling_rejected(self, root: Path) -> None:
        """A folder next to the project is rejected."""
        with pytest.raises(CorpusError, match="outside the project root"):
            CorpusReader(root.parent / "elsewhere").read()

    def test_parent_rejected_even_if_files_inside(
            self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The parent folder is rejected even when every readable file
        found in it lives inside the project (the ``--raw_dir ../`` bug).
        """
        (root.parent / "elsewhere" / "b.md").unlink()
        with pytest.raises(CorpusError, match="outside the project root"):
            CorpusReader(root.parent).read()

    def test_relative_parent_rejected(
            self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """``../`` launched from the project root is rejected."""
        monkeypatch.chdir(root)
        with pytest.raises(CorpusError, match="outside the project root"):
            CorpusReader("../").read()

    def test_rejected_before_listing(
            self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """No file is listed or read when the folder is outside."""
        def must_not_run(self: CorpusReader) -> list[Path]:
            """Fail the test if files are listed."""
            raise AssertionError("files were listed")
        monkeypatch.setattr(CorpusReader, "_list_files", must_not_run)
        with pytest.raises(CorpusError, match="outside the project root"):
            CorpusReader(root.parent).read()

    def test_message_names_both_folders(self, root: Path) -> None:
        """The message says which folder and which root."""
        with pytest.raises(CorpusError) as info:
            CorpusReader(root.parent).read()
        assert str(root.parent) in str(info.value)
        assert str(root) in str(info.value)

    def test_not_found_checked_first(self, root: Path) -> None:
        """A missing folder outside the project is reported as missing."""
        with pytest.raises(CorpusError, match="Corpus folder not found"):
            CorpusReader(root.parent / "nope").read()

    def test_subfolder_accepted(self, root: Path) -> None:
        """Any folder inside the project is fine."""
        result = CorpusReader(root / "data" / "raw").read()
        assert [d.file_path for d in result.documents] == ["data/raw/a.md"]

    def test_project_root_accepted(self, root: Path) -> None:
        """The project root itself is inside (reading it is the user's
        choice, even though it would include ``.venv``).
        """
        result = CorpusReader(root).read()
        assert [d.file_path for d in result.documents] == ["data/raw/a.md"]

    def test_relative_path_safety_net(self, root: Path) -> None:
        """``_relative_path`` still rejects a file outside the root and
        chains the original ``ValueError``.
        """
        outside_file = root.parent / "elsewhere" / "b.md"
        with pytest.raises(CorpusError) as info:
            CorpusReader(root)._relative_path(outside_file)
        assert isinstance(info.value.__cause__, ValueError)


@pytest.fixture(scope="module")
def result() -> CorpusReadResult:
    """Read the real corpus once for the whole module."""
    return CorpusReader(REAL_RAW).read()


@pytest.mark.skipif(not REAL_RAW.is_dir(), reason="vLLM corpus not present")
class TestRealCorpus:
    """Checks against the real vLLM corpus in ``data/raw``."""

    def test_every_file_read(self, result: CorpusReadResult) -> None:
        """All candidate files are valid UTF-8."""
        assert result.skipped == []
        assert result.found == len(result.documents)
        assert result.found > 1000

    def test_paths_prefix(self, result: CorpusReadResult) -> None:
        """Every path starts like the moulinette's paths."""
        assert all(d.file_path.startswith("data/raw/vllm-0.10.1/")
                   for d in result.documents)

    @pytest.mark.skipif(not DATASETS.is_dir(), reason="datasets not present")
    def test_dataset_sources_present(self, result: CorpusReadResult) -> None:
        """Every ground-truth source path is one of the read documents."""
        paths = {d.file_path for d in result.documents}
        missing = []
        for dataset in (DATASETS / "AnsweredQuestions").glob("*.json"):
            data = json.loads(dataset.read_text(encoding="utf-8"))
            for question in data["rag_questions"]:
                for source in question["sources"]:
                    if source["file_path"] not in paths:
                        missing.append(source["file_path"])
        assert missing == []

    @pytest.mark.skipif(not DATASETS.is_dir(), reason="datasets not present")
    def test_dataset_offsets(self, result: CorpusReadResult) -> None:
        """A known ground-truth span points at the expected text."""
        docs = {d.file_path: d.text for d in result.documents}
        text = docs["data/raw/vllm-0.10.1/docs/features/lora.md"]
        assert text[4695:4718] == "### Using API Endpoints"
