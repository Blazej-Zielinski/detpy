import numpy as np
import copy

from detpy.models.population import Population


class BinomialCrossover:
    """
    Binomial crossover operator for your Member & Population classes.
    Assumes:
        member.chromosomes[i].real_value exists
        member.args_num gives dimension
    """

    def __init__(self, ensure_at_least_one=True):
        self.ensure_at_least_one = ensure_at_least_one

    def crossover_members(self, target_member, mutant_member, cr: float):
        new_member = copy.deepcopy(target_member)
        D = new_member.args_num

        mask = np.random.rand(D) <= cr

        if self.ensure_at_least_one:
            j_rand = np.random.randint(0, D)
            mask[j_rand] = True

        for i in range(D):
            if mask[i]:
                new_member.chromosomes[i].real_value = mutant_member.chromosomes[i].real_value
            else:
                new_member.chromosomes[i].real_value = target_member.chromosomes[i].real_value

        return new_member

    def crossover_population(self, origin_population, mutated_population, cr_table):
        new_members = [
            self.crossover_members(origin_population.members[i],
                                   mutated_population.members[i],
                                   cr_table[i])
            for i in range(origin_population.size)
        ]

        return Population.with_new_members(origin_population, new_members)
