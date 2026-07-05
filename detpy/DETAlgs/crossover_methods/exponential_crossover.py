import copy
import numpy as np

from detpy.models.population import Population


class ExponentialCrossover:
    """
    Exponential crossover operator for Member & Population classes.
    """
    def crossover_members(self, target_member, mutant_member, cr: float):
        new_member = copy.deepcopy(target_member)

        d = new_member.args_num

        start = np.random.randint(d)

        l = 1
        while l < d and np.random.rand() < cr:
            l += 1

        for k in range(l):
            idx = (start + k) % d
            new_member.chromosomes[idx].real_value = (
                mutant_member.chromosomes[idx].real_value
            )

        return new_member

    def crossover_population(self, origin_population,
                             mutated_population,
                             cr_table):

        new_members = [
            self.crossover_members(
                origin_population.members[i],
                mutated_population.members[i],
                cr_table[i]
            )
            for i in range(origin_population.size)
        ]

        return Population.with_new_members(origin_population, new_members)