from random import randrange

from detpy.models.member import Member


class ArchiveReduction:
    def reduce_archive(
            self,
            archive: list[Member],
            archive_size: int,
    ) -> list[Member]:
        """
        Randomly remove archive members until the archive
        does not exceed the specified maximum size.

        Parameters:
        - archive (list[Member]): The archive of members from previous populations.
        - archive_size (int): The desired size of the archive.
        """
        max_size = max(0, archive_size)

        while len(archive) > max_size:
            idx = randrange(len(archive))
            archive.pop(idx)

        return archive
