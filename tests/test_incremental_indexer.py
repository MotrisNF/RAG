"""Tests for the incremental indexing (bonus 3).

The models that describe the corpus (``FileStamp``, ``IndexManifest``,
``CorpusChanges``) and ``IncrementalIndexer._diff``, which compares the
saved manifest with the files found now. Nothing here touches the disk:
the "files" are dictionaries of stamps.
"""
import pytest
from pydantic import ValidationError

from src.incremental_indexer import IncrementalIndexer
from src.models import CorpusChanges, FileStamp, IndexManifest

OLD = FileStamp(mtime_ns=1_000, size=10)
NEWER = FileStamp(mtime_ns=2_000, size=10)
BIGGER = FileStamp(mtime_ns=1_000, size=11)


def no_changes() -> CorpusChanges:
    """Three empty lists."""
    return CorpusChanges(new=[], modified=[], deleted=[])


class TestFileStamp:
    """When a file last changed."""

    def test_equal(self) -> None:
        """Same time and size: the same stamp (the file did not change)."""
        assert FileStamp(mtime_ns=1_000, size=10) == OLD

    @pytest.mark.parametrize("other", [NEWER, BIGGER])
    def test_different(self, other: FileStamp) -> None:
        """Another time or another size: another stamp."""
        assert other != OLD

    def test_nanoseconds_kept(self) -> None:
        """``st_mtime_ns`` values (about 1.8e18) are kept exactly: two
        writes 1 ns apart are told apart."""
        stamp = FileStamp(mtime_ns=1_791_369_129_831_588_221, size=1)
        twin = FileStamp(mtime_ns=1_791_369_129_831_588_222, size=1)
        assert stamp != twin
        assert FileStamp.model_validate_json(
            stamp.model_dump_json()) == stamp

    @pytest.mark.parametrize("missing", ["mtime_ns", "size"])
    def test_field_required(self, missing: str) -> None:
        """Both fields are required."""
        data = {"mtime_ns": 1, "size": 1}
        del data[missing]
        with pytest.raises(ValidationError):
            FileStamp.model_validate(data)


class TestIndexManifest:
    """What ``manifest.json`` holds."""

    @pytest.fixture
    def manifest(self) -> IndexManifest:
        """A manifest with two files."""
        return IndexManifest(max_chunk_size=2000, raw_dir="data/raw",
                             files={"data/raw/a.py": OLD,
                                    "data/raw/b.md": NEWER})

    def test_round_trip(self, manifest: IndexManifest) -> None:
        """Saved as JSON and read back, it is the same manifest."""
        text = manifest.model_dump_json()
        assert IndexManifest.model_validate_json(text) == manifest

    def test_no_files(self) -> None:
        """An empty corpus is a valid manifest."""
        manifest = IndexManifest(max_chunk_size=500, raw_dir="x", files={})
        assert manifest.files == {}

    @pytest.mark.parametrize("size", [0, -1, 2001])
    def test_chunk_size_out_of_range(self, size: int) -> None:
        """``max_chunk_size`` has the limits of ``index`` (1 to 2000)."""
        with pytest.raises(ValidationError):
            IndexManifest(max_chunk_size=size, raw_dir="data/raw",
                          files={})

    @pytest.mark.parametrize("text", [
        "",
        "not json",
        "[]",
        '{"raw_dir": "data/raw", "files": {}}',
        '{"max_chunk_size": 2000, "files": {}}',
        '{"max_chunk_size": 2000, "raw_dir": "data/raw"}',
        '{"max_chunk_size": 2000, "raw_dir": "data/raw",'
        ' "files": {"a.py": {"size": 1}}}',
        '{"max_chunk_size": 2000, "raw_dir": "data/raw",'
        ' "files": {"a.py": {"mtime_ns": "soon", "size": 1}}}',
        '{"max_chunk_size": 2000, "raw_dir": "data/raw", "files": []}',
    ])
    def test_broken_rejected(self, text: str) -> None:
        """A broken or incomplete manifest is a ``ValidationError``
        (``index`` will treat it as no manifest: a full rebuild)."""
        with pytest.raises(ValidationError):
            IndexManifest.model_validate_json(text)


class TestCorpusChanges:
    """The three lists and the questions they answer."""

    @pytest.fixture
    def changes(self) -> CorpusChanges:
        """One file of each kind."""
        return CorpusChanges(new=["w.md"], modified=["y.py"],
                             deleted=["z.md"])

    def test_to_remove(self, changes: CorpusChanges) -> None:
        """Modified and deleted files lose their old chunks."""
        assert changes.to_remove == ["y.py", "z.md"]

    def test_to_read(self, changes: CorpusChanges) -> None:
        """New and modified files are read and chunked."""
        assert changes.to_read == ["w.md", "y.py"]

    def test_modified_in_both(self, changes: CorpusChanges) -> None:
        """A modified file is removed and read again."""
        assert "y.py" in changes.to_remove
        assert "y.py" in changes.to_read

    def test_sorted(self) -> None:
        """Both lists are sorted, whatever list a path comes from."""
        changes = CorpusChanges(new=["b", "d"], modified=["c", "a"],
                                deleted=["e", "0"])
        assert changes.to_read == ["a", "b", "c", "d"]
        assert changes.to_remove == ["0", "a", "c", "e"]

    def test_empty(self) -> None:
        """No list with anything: nothing changed."""
        assert no_changes().is_empty
        assert no_changes().to_read == []
        assert no_changes().to_remove == []

    @pytest.mark.parametrize("field", ["new", "modified", "deleted"])
    def test_one_change_not_empty(self, field: str) -> None:
        """A single file in any list is a change."""
        changes = no_changes().model_copy(update={field: ["a.py"]})
        assert not changes.is_empty


class TestDiff:
    """``IncrementalIndexer._diff``: saved stamps against current ones."""

    def test_nothing_changed(self) -> None:
        """Same files, same stamps: no change."""
        files = {"a.py": OLD, "b.md": NEWER}
        assert IncrementalIndexer._diff(files, dict(files)).is_empty

    def test_both_empty(self) -> None:
        """No files before nor now: no change."""
        assert IncrementalIndexer._diff({}, {}).is_empty

    def test_new_file(self) -> None:
        """A path only in the current files is new."""
        changes = IncrementalIndexer._diff({"a.py": OLD},
                                           {"a.py": OLD, "b.py": OLD})
        assert changes == CorpusChanges(new=["b.py"], modified=[],
                                        deleted=[])

    def test_deleted_file(self) -> None:
        """A path only in the saved files was deleted."""
        changes = IncrementalIndexer._diff({"a.py": OLD, "b.py": OLD},
                                           {"a.py": OLD})
        assert changes == CorpusChanges(new=[], modified=[],
                                        deleted=["b.py"])

    @pytest.mark.parametrize("now", [NEWER, BIGGER])
    def test_modified_file(self, now: FileStamp) -> None:
        """Same path with another time or size: modified."""
        changes = IncrementalIndexer._diff({"a.py": OLD}, {"a.py": now})
        assert changes == CorpusChanges(new=[], modified=["a.py"],
                                        deleted=[])

    def test_unchanged_not_listed(self) -> None:
        """Files with the same stamp are in no list."""
        changes = IncrementalIndexer._diff({"a.py": OLD, "b.py": OLD},
                                           {"a.py": OLD, "b.py": NEWER})
        assert "a.py" not in (changes.new + changes.modified
                              + changes.deleted)

    def test_all_at_once(self) -> None:
        """New, modified, deleted and unchanged files together."""
        old = {"keep.py": OLD, "edit.py": OLD, "gone.md": OLD}
        new = {"keep.py": OLD, "edit.py": NEWER, "added.md": OLD}
        changes = IncrementalIndexer._diff(old, new)
        assert changes == CorpusChanges(new=["added.md"],
                                        modified=["edit.py"],
                                        deleted=["gone.md"])
        assert changes.to_remove == ["edit.py", "gone.md"]
        assert changes.to_read == ["added.md", "edit.py"]

    def test_no_manifest_all_new(self) -> None:
        """No saved files (first ``index``): every file is new, so a full
        index is the incremental case with an empty past."""
        new = {"b.py": OLD, "a.md": OLD, "c.txt": OLD}
        changes = IncrementalIndexer._diff({}, new)
        assert changes == CorpusChanges(new=["a.md", "b.py", "c.txt"],
                                        modified=[], deleted=[])

    def test_corpus_emptied(self) -> None:
        """Every saved file gone: every file is deleted."""
        changes = IncrementalIndexer._diff({"a.py": OLD, "b.md": OLD}, {})
        assert changes.deleted == ["a.py", "b.md"]
        assert changes.to_read == []

    def test_lists_sorted(self) -> None:
        """Each list is sorted, whatever the order of the dictionaries
        (sets have no fixed order)."""
        paths = [f"data/raw/{name}.py" for name in "qwertyuiop"]
        changes = IncrementalIndexer._diff(
            {}, {path: OLD for path in paths})
        assert changes.new == sorted(paths)

    def test_inputs_unchanged(self) -> None:
        """The two dictionaries are only read."""
        old = {"a.py": OLD, "b.py": OLD}
        new = {"a.py": NEWER, "c.py": OLD}
        old_copy, new_copy = dict(old), dict(new)
        IncrementalIndexer._diff(old, new)
        assert old == old_copy
        assert new == new_copy
