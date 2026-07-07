from detpy.DETAlgs.random.index_generator import IndexGenerator
from detpy.models.enums.optimization import OptimizationType

import copy

from detpy.models.member import Member
from detpy.models.population import Population


class MutationRandrl1:
    """
    Implements the 'randrl/1' mutation method.

    Formula:
        rx1 + F * (rx2 - rx3)

    where rx1 is the best individual among three randomly selected members.
    """

    @staticmethod
    def mutate(
            population: Population,
            current_index: int,
            f: float,
            optimization: OptimizationType
    ) -> Member:
        index_gen = IndexGenerator()

        indices = index_gen.generate_unique_multiple(
            population.size,
            3,
            [current_index]
        )

        candidates = [population.members[idx] for idx in indices]

        if optimization == OptimizationType.MINIMIZATION:
            best_random = min(candidates, key=lambda m: m.fitness_value)
        else:
            best_random = max(candidates, key=lambda m: m.fitness_value)

        others = [m for m in candidates if m is not best_random]

        r2, r3 = others

        mutant = copy.deepcopy(best_random)
        mutant.chromosomes = (
                best_random.chromosomes
                + f * (r2.chromosomes - r3.chromosomes)
        )

        return mutant
