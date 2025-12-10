import copy
from random import randrange

import numpy as np

from detpy.models.member import Member
from detpy.models.population import Population


def archive_reduction(archive: list[Member], archive_size: int, pop_size: int):
    """
    Reduce the size of the archive to the specified size.

    Parameters:
    - archive (list[Member]): The archive of members from previous populations.
    - archive_size (int): The desired size of the archive.
    - pop_size (int): The size of the population.

    Returns: The reduced archive.
    """
    if archive_size == 0:
        archive.clear()

    max_elem = min(archive_size, pop_size)

    reduce_num = len(archive) - max_elem
    for _ in range(reduce_num):
        idx = randrange(len(archive))
        archive.pop(idx)

    return archive