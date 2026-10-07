"""Incremental indexing (bonus 3): re-index only the files that changed.

The saved ``manifest.json`` records the stamp (modification time and
size) of every corpus file. The next ``index`` compares it with the
current files and only reads, chunks, tokenizes and embeds the new and
modified ones; the chunks of modified and deleted files leave both
indexes.
"""
from .models import Chunk, CorpusChanges, FileStamp, IndexPlan


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

    @staticmethod
    def _plan(old_chunks: list[Chunk],
              to_remove: list[str],
              new_chunks: list[Chunk]) -> IndexPlan:
        """Decide the chunks of the updated index and renumber the kept
        ones, without touching the indexes.

        ``order_key`` identifies a chunk: a kept chunk and a new one never
        share it, because new chunks only come from new or modified files
        and the old chunks of those are removed.

        Args:
            old_chunks: The chunks of the saved index, by chunk id.
            to_remove: Files whose old chunks leave the index
                (``CorpusChanges.to_remove``).
            new_chunks: The chunks of the files read again
                (``CorpusChanges.to_read``), in any order.

        Returns:
            The kept and new chunks sorted as a full index would sort
            them, and ``{old id: new id}`` of the kept ones.
        """
        removed = set(to_remove)
        old_ids = {chunk.order_key(): old_id
                   for old_id, chunk in enumerate(old_chunks)
                   if chunk.file_path not in removed}
        kept = [chunk for chunk in old_chunks
                if chunk.file_path not in removed]
        chunks = sorted(
            new_chunks + kept,
            key=Chunk.order_key
        )
        remap = {old_ids[chunk.order_key()]: new_id
                 for new_id, chunk in enumerate(chunks)
                 if chunk.order_key() in old_ids}
        return IndexPlan(chunks=chunks, remap=remap)
