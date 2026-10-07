"""Incremental indexing (bonus 3): re-index only the files that changed.

The saved ``manifest.json`` records the stamp (modification time and
size) of every corpus file. The next ``index`` compares it with the
current files and only reads, chunks, tokenizes and embeds the new and
modified ones; the chunks of modified and deleted files leave both
indexes.
"""
from .models import CorpusChanges, FileStamp


class IncrementalIndexer:
    """Bridge between the corpus reader and the two indexes.

    Decides what changed since the saved index and updates the indexes
    with only those files.
    """

    @staticmethod
    def _diff(old: dict[str, FileStamp],
              new: dict[str, FileStamp]) -> CorpusChanges:
        """Compare the saved stamps with the current ones.

        Static: it only needs its two arguments, so it is tested without
        files or an index.

        Args:
            old: Stamps from the saved manifest (empty when there is
                none: then every file is new).
            new: Stamps of the corpus now (``CorpusReader.stamps``).

        Returns:
            New paths (only in ``new``), deleted paths (only in ``old``)
            and modified paths (in both, with another stamp), each list
            sorted. Unchanged files are in no list.
        """
        news = new.keys() - old.keys()
        deleted = old.keys() - new.keys()
        common = old.keys() & new.keys()
        modified = [file for file in common if old[file] != new[file]]
        return CorpusChanges(
            new=sorted(news),
            deleted=sorted(deleted),
            modified=sorted(modified)
        )
