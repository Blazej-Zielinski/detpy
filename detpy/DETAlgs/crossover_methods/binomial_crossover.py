import numpy as np
import copy

from detpy.models.population import Population


class BinomialCrossover:
    """
    Binomial crossover operator for Member & Population classes.
    """

    def __init__(self, ensure_at_least_one=True):
        """
        Initialize the BinomialCrossover operator.

        Parameters:
        - ensure_at_least_one (bool): If True, ensures at least one gene is crossed over.
        """
        self.ensure_at_least_one = ensure_at_least_one

    def crossover_members(self, target_member, mutant_member, cr: float):
        """
        Perform crossover between a target member and a mutant member.

        Parameters:
        - target_member (Member): The target member from the original population.
        - mutant_member (Member): The mutant member generated during mutation.
        - cr (float): The crossover rate, determining the probability of gene exchange.

        Returns:
        - Member: A new member created by combining genes from the target and mutant members.
        """
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
        """
        Perform crossover operation for the entire population.

        Parameters:
        - origin_population (Population): The original population before crossover.
        - mutated_population (Population): The population after mutation.
        - cr_table (List[float]): List of crossover rates for each individual.

        Returns:
        - Population: A new population created by applying crossover to the original and mutated populations.
        """
        new_members = [
            self.crossover_members(origin_population.members[i],
                                   mutated_population.members[i],
                                   cr_table[i])
            for i in range(origin_population.size)
        ]

        return Population.with_new_members(origin_population, new_members)
