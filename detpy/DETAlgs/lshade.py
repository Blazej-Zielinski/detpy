import copy
from typing import List

import numpy as np

from detpy.DETAlgs.archive_reduction.archive_reduction import ArchiveReduction
from detpy.DETAlgs.base import BaseAlg
from detpy.DETAlgs.crossover_methods.binomial_crossover import BinomialCrossover
from detpy.DETAlgs.data.alg_data import LShadeData
from detpy.DETAlgs.math.math_functions import MathFunctions
from detpy.DETAlgs.mutation_methods.current_to_pbest_1 import MutationCurrentToPBest1
from detpy.DETAlgs.random.index_generator import IndexGenerator
from detpy.DETAlgs.random.random_value_generator import RandomValueGenerator
from detpy.models.enums.boundary_constrain import fix_boundary_constraints_with_parent
from detpy.models.enums.optimization import OptimizationType

from detpy.models.population import Population


class LSHADE(BaseAlg):
    """
        LSHADE: Improving the Search Performance of SHADE Using Linear Population Size Reduction

        Links:
        https://ieeexplore.ieee.org/document/6900380

        References:
        Tanabe, R., & Fukunaga, A. S. (2014). Improving the search performance of SHADE using linear population size
        reduction. In 2014 IEEE Congress on Evolutionary Computation (CEC) (pp. 1658–1665). 2014 IEEE Congress on
        Evolutionary Computation (CEC). IEEE. https://doi.org/10.1109/cec.2014.6900380
    """

    def __init__(self, params: LShadeData, db_conn=None, db_auto_write=False):
        super().__init__(LSHADE.__name__, params, db_conn, db_auto_write)

        self._H = params.memory_size  # Memory size for f and cr adaptation
        self._memory_F = np.full(self._H, 0.5)  # Initial memory for F
        self._memory_Cr = np.full(self._H, 0.5)  # Initial memory for Cr
        self._p = params.best_member_percentage
        self._k_index = 0

        self._successCr = []
        self._successF = []
        self._difference_fitness_success = []

        self._min_the_best_percentage = 2 / self.population_size  # Minimal percentage of the best members to consider

        self._archive_size = self.population_size  # Size of the archive
        self._archive = []  # Archive for storing the members from old populations

        self._min_pop_size = params.minimum_population_size  # Minimal population size

        self._start_population_size = self.population_size

        self._population_size_reduction_strategy = params.population_reduction_strategy

        # Define a terminal value for memory_Cr. It indicates that no successful Cr was found in the epoch.
        self._TERMINAL = np.nan

        # We need this value for checking close to zero in update_memory
        self._EPSILON = 0.00001

        self._index_gen = IndexGenerator()
        self._random_value_gen = RandomValueGenerator()
        self._binomial_crossing = BinomialCrossover()
        self._archive_reduction = ArchiveReduction()

    def update_population_size(self, nfe: int, total_nfe: int, start_pop_size: int, min_pop_size: int):
        """
        Calculate new population size using Linear Population Size Reduction (LPSR).

        Parameters:
        - nfe (int): The current function evaluations.
        - total_nfe (int): The total number of function evaluations.
        - start_pop_size (int): The initial population size.
        - min_pop_size (int): The minimum population size.
        """
        new_size = self._population_size_reduction_strategy.get_new_population_size(
            nfe, total_nfe, start_pop_size, min_pop_size
        )
        self._pop.resize(new_size)

    def mutate(self,
               population: Population,
               the_best_to_select_table: List[int],
               f_table: List[float]
               ) -> Population:
        """
        Perform mutation step for the population in SHADE.

        Parameters:
        - population (Population): The population to mutate.
        - the_best_to_select_table (List[int]): List of the number of the best members to select.
        - f_table (List[float]): List of scaling factors for mutation.

        Returns: A new Population with mutated members.
        """
        new_members = []

        sum_archive_and_population = np.concatenate((population.members, self._archive))

        for i in range(population.size):
            r1 = self._index_gen.generate_unique(len(population.members), [i])

            # Archive is included population and archive members
            r2 = self._index_gen.generate_unique(len(sum_archive_and_population), [i, r1])

            # Select top p-best members from the population
            best_members = population.get_best_members(the_best_to_select_table[i])

            # Randomly select one of the p-best members
            random_index = self._index_gen.generate(0, len(best_members))
            selected_best_member = best_members[random_index]

            # Apply the mutation formula (current-to-pbest strategy)
            mutated_member = MutationCurrentToPBest1.mutate(
                base_member=population.members[i],
                best_member=selected_best_member,
                r1=population.members[r1],
                r2=sum_archive_and_population[r2],
                f=f_table[i]
            )

            new_members.append(mutated_member)

        # Create a new population with the mutated members
        new_population = Population.with_new_members(population, new_members)
        return new_population

    def _selection(self, origin_population, modified_population, ftable, cr_table):
        """
        Perform the selection step for the SHADE algorithm.

        This method selects the members for the next generation based on their fitness values.
        If the modified member is better than the original member, it is selected; otherwise, the original member is retained.
        Additionally, information about successful trials is stored for updating memory parameters.

        Parameters:
        - origin_population (Population): The original population before the selection step.
        - modified_population (Population): The modified population after mutation and crossover.
        - ftable (List[float]): List of scaling factors used during mutation.
        - cr_table (List[float]): List of crossover rates used during crossover.

        Returns:
        - Population: A new population containing the selected members for the next generation.
        """
        optimization = origin_population.optimization
        new_members = []

        # Define comparison and difference functions based on optimization type
        if optimization == OptimizationType.MINIMIZATION:
            is_better = lambda orig, mod: mod.fitness_value < orig.fitness_value
            diff = lambda orig, mod: orig.fitness_value - mod.fitness_value
        else:  # MAXIMIZATION
            is_better = lambda orig, mod: mod.fitness_value > orig.fitness_value
            diff = lambda orig, mod: mod.fitness_value - orig.fitness_value

        # Iterate through the population and perform selection
        for i in range(origin_population.size):
            orig = origin_population.members[i]
            mod = modified_population.members[i]

            if not is_better(orig, mod):
                # Retain the original member if it is better or equal
                new_members.append(copy.deepcopy(orig))
                continue

            # If the modified member is better, update the archive and success metrics
            self._archive.append(copy.deepcopy(orig))
            self._successF.append(ftable[i])
            self._successCr.append(cr_table[i])
            self._difference_fitness_success.append(diff(orig, mod))
            new_members.append(copy.deepcopy(mod))

        # Return a new population with the selected members
        return Population.with_new_members(origin_population, new_members)

    def update_memory(self, success_f: List[float], success_cr: List[float], difference_fitness_success: List[float]):
        """
        Update the memory for the crossover rates and scaling factors based on the success of the trial vectors.

        Parameters:
        - success_f (List[float]): List of scaling factors that led to better trial vectors.
        - success_cr (List[float]): List of crossover rates that led to better trial vectors.
        - difference_fitness_success (List[float]): List of differences in objective function values (|f(u_k, G) - f(x_k, G)|).
        """
        if len(success_f) > 0 and len(success_cr) > 0:
            total = np.sum(difference_fitness_success)
            weights = difference_fitness_success / total

            if np.isclose(total, 0.0, atol=self._EPSILON):
                self._memory_Cr[self._k_index] = self._TERMINAL

            else:
                cr_new = MathFunctions.calculate_lehmer_mean(np.array(success_cr), weights, p=2)
                cr_new = np.clip(cr_new, 0, 1)
                self._memory_Cr[self._k_index] = cr_new

            f_new = MathFunctions.calculate_lehmer_mean(np.array(success_f), weights, p=2)
            f_new = np.clip(f_new, 0, 1)
            self._memory_F[self._k_index] = f_new

            # Reset the lists for the next generation
            self._successF = []
            self._successCr = []
            self._difference_fitness_success = []
            self._k_index = (self._k_index + 1) % self._H

    def initialize_parameters_for_epoch(self):
        """
        Initialize the parameters for the next epoch of the SHADE algorithm.
        f_table: List of scaling factors for mutation.
        cr_table: List of crossover rates.
        the_bests_to_select: List of the number of the best members to select because in crossover we need to select
        the best members from the population for one factor.

        Returns:
        - f_table (List[float]): List of scaling factors for mutation.
        - cr_table (List[float]): List of crossover rates.
        - the_bests_to_select (List[int]): List of the number of the best members to possibly select.
        """
        f_table = []
        cr_table = []
        the_bests_to_select = []

        for i in range(self._pop.size):
            ri = np.random.randint(0, self._H)

            # We check here if we have TERMINAL value in memory_Cr
            if np.isnan(self._memory_Cr[ri]):
                cr = 0.0
            else:
                cr = self._random_value_gen.generate_normal(self._memory_Cr[ri], 0.1, 0.0, 1.0)

            mean_f = self._memory_F[ri]

            f = self._random_value_gen.generate_cauchy_greater_than_zero(mean_f, 0.1, 0.0, 1.0)

            f_table.append(f)
            cr_table.append(cr)

            the_best_to_possible_select = int(self.population_size * self._p)

            the_bests_to_select.append(the_best_to_possible_select)

        return f_table, cr_table, the_bests_to_select

    def next_epoch(self):
        """
        Perform the next epoch of the SHADE algorithm.
        """
        f_table, cr_table, the_bests_to_select = self.initialize_parameters_for_epoch()

        mutant = self.mutate(self._pop, the_bests_to_select, f_table)

        # Crossover step
        trial = self._binomial_crossing.crossover_population(self._pop, mutant, cr_table)

        fix_boundary_constraints_with_parent(self._pop, trial, self.boundary_constraints_fun)

        # Evaluate fitness values for the trial population
        trial.update_fitness_values(self._function.eval, self.parallel_processing)

        # Selection step
        new_pop = self._selection(self._pop, trial, f_table, cr_table)

        # Archive management
        self._archive_size = self.population_size

        self._archive = self._archive_reduction.reduce_archive(self._archive, self._archive_size, self.population_size)

        # Update the population
        self._pop = new_pop

        # Update the memory for CR and F
        self.update_memory(self._successF, self._successCr, self._difference_fitness_success)

        self.update_population_size(self.nfe, self.nfe_max, self._start_population_size,
                                    self._min_pop_size)
