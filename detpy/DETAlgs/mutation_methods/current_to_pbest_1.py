import copy

from detpy.models.member import Member


class MutationCurrentToPBest1:
    """
    Implements the 'current-to-pbest/1' mutation method.

    Formula: bm + F * (bm_best - bm) + F * (r1 - r2)
    """

    @staticmethod
    def mutate(base_member: Member, best_member: Member, r1: Member, r2: Member, f: float) -> Member:
        """
        Performs the mutation operation.

        Parameters:
        - base_member (Member): The base member used for the mutation operation.
        - best_member (Member): The best member, typically the one with the best fitness value, used in the mutation formula.
        - r1 (Member): A randomly selected member from the population, used for mutation. (rank selection)
        - r2 (Member): Another randomly selected member from the archive, used for mutation.
        - f (float): A scaling factor that controls the magnitude of the mutation between random members of the population.

        Returns:
        - Member: A new member with the mutated chromosomes.
        """
        new_member = copy.deepcopy(base_member)
        new_member.chromosomes = base_member.chromosomes + (
                f * (best_member.chromosomes - base_member.chromosomes)) + (
                                         f * (r1.chromosomes - r2.chromosomes))
        return new_member
