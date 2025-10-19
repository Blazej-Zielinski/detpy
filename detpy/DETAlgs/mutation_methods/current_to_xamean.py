import copy

import numpy as np

from detpy.DETAlgs.methods.methods_alshade import wrap_as_chromosomes
from detpy.models.member import Member


class MutationCurrentToXamean:
    """
    Implements the current-to-x_Amean/1 mutation strategy:

    Formula:    v_i = x_i + F * (x_Amean - x_i) + F * (x_r1 - x_r2)
    """

    @staticmethod
    def mutate(base_member: Member, xamean: np.ndarray, r1: Member, r2: Member, f: float) -> Member:
        """
        Parameters:
        - base_member (Member): The base member used for the mutation operation.
        - xamean (np.ndarray): Estimation of the global optimal solution based on the promising members in the population.
        - r1 (Member): A randomly selected member from the population based on rank selection, used for mutation.
        - r2 (Member): Another randomly selected member from the population based on rank selection, used for mutation.
        - f (float): A scaling factor.

        Returns: A new member with the mutated chromosomes.
        """
        xamean_chromosomes = wrap_as_chromosomes(xamean, base_member.chromosomes)
        xamean_as_member = copy.deepcopy(base_member)
        xamean_as_member.chromosomes = xamean_chromosomes

        new_member = copy.deepcopy(base_member)
        new_member.chromosomes = (
                base_member.chromosomes
                + f * (xamean_as_member.chromosomes - base_member.chromosomes)
                + f * (r1.chromosomes - r2.chromosomes)
        )
        return new_member
