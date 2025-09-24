import copy

from detpy.models.member import Member


class MutationCurrentToPBestR:
    """
    Implements the 'current-to-pbest/r' mutation method.

    Formula: bm + Fw * (bm_best - bm) + F * (r1 - r2)
    """

    @staticmethod
    def mutate(base_member: Member, best_member: Member, r1: Member, r2: Member, f: float, fw: float):
        """
        Formula: bm + Fw * (bm_best - bm) + F * (r1 - r2)

        Parameters:
        - base_member (Member): The base member used for the mutation operation.
        - best_member (Member): The best member, typically the one with the best fitness value, used in the mutation formula.
        - r1 (Member): A randomly selected member from the population based on rank selection, used for mutation.
        - r2 (Member): Another randomly selected member from the population based on rank selection, used for mutation.
        - f (float): A scaling factor for the first part equation.
        - fw (float): A scaling factor for the second part equation.

        Returns: A new member with the mutated chromosomes.
        """
        new_member = copy.deepcopy(base_member)
        new_member.chromosomes = base_member.chromosomes + (
                fw * (best_member.chromosomes - base_member.chromosomes)) + (
                                         f * (r1.chromosomes - r2.chromosomes))
        return new_member
