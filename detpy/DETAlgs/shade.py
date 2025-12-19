import copy
from typing import List

import numpy as np

from detpy.DETAlgs.archive_reduction.archive_reduction import ArchiveReduction
from detpy.DETAlgs.base import BaseAlg
from detpy.DETAlgs.crossover_methods.binomial_crossover import BinomialCrossover
from detpy.DETAlgs.data.alg_data import ShadeData
from detpy.DETAlgs.math.math_functions import MathFunctions
from detpy.DETAlgs.mutation_methods.current_to_pbest_1 import MutationCurrentToPBest1
from detpy.DETAlgs.random.index_generator import IndexGenerator
from detpy.DETAlgs.random.random_value_generator import RandomValueGenerator

from detpy.models.enums.boundary_constrain import fix_boundary_constraints
from detpy.models.enums.optimization import OptimizationType

from detpy.models.population import Population


class SHADE(BaseAlg):
    """
        SHADE: Success-History based Adaptive Differential Evolution

        References:
        Ryoji Tanabe and Alex Fukunaga Graduate School of Arts and Sciences The University of Tokyo
    """

    def __init__(self, params: ShadeData, db_conn=None, db_auto_write=False):
        super().__init__(SHADE.__name__, params, db_conn, db_auto_write)

        self._H = params.memory_size  # Memory size for f and cr adaptation
        self._memory_F = np.full(self._H, 0.5)  # Initial memory for F
        self._memory_Cr = np.full(self._H, 0.5)  # Initial memory for Cr

        self._successCr = []
        self._successF = []
        self._difference_fitness_success = []

        self._min_the_best_percentage = 2 / self.population_size  # Minimal percentage of the best members to consider

        self._archive_size = self.population_size  # Size of the archive is the same as population
        self._archive = []  # Archive for storing the members from old populations

        self._k_index = 0

        self._index_gen = IndexGenerator()
        self._random_value_gen = RandomValueGenerator()
        self._binomial_crossing = BinomialCrossover()
        self._archive_reduction = ArchiveReduction()

    def _mutate(self,
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

    def _update_memory(self, success_f: List[float], success_cr: List[float], difference_fitness_success: List[float]):
        """
        Update the memory for the crossover rates and scaling factors based on the success of the trial vectors.

        Parameters:
        - success_f (List[float]): List of scaling factors that led to better trial vectors.
        - success_cr (List[float]): List of crossover rates that led to better trial vectors.
        - difference_fitness_success (List[float]): List of differences in objective function values (|f(u_k, G) - f(x_k, G)|).
        """
        if len(success_f) > 0 and len(success_cr) > 0:
            total = np.sum(difference_fitness_success)
            if total == 0:
                return

            weights = difference_fitness_success / total
            f_new = MathFunctions.calculate_lehmer_mean(np.array(success_f), weights, p=2)
            f_new = np.clip(f_new, 0, 1)

            cr_new = MathFunctions.calculate_weighted_average(np.array(success_cr), weights)
            cr_new = np.clip(cr_new, 0, 1)

            self._memory_F[self._k_index] = f_new
            self._memory_Cr[self._k_index] = cr_new

            # Reset the lists for the next generation
            self._successF = []
            self._successCr = []
            self._difference_fitness_success = []
            self._k_index = (self._k_index + 1) % self._H

    def _initialize_parameters_for_epoch(self):
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
            ri = self._index_gen.generate(0, self._H)
            mean_f = self._memory_F[ri]
            mean_cr = self._memory_Cr[ri]
            cr = self._random_value_gen.generate_normal(mean_cr, 0.1, 0.0, 1.0)

            f = self._random_value_gen.generate_cauchy_greater_than_zero(mean_f, 0.1, 0.0, 1.0)

            f_table.append(f)
            cr_table.append(cr)

            min_percentage = 2 / self.population_size
            max_percentage = 0.2

            the_best_to_possible_select = self._random_value_gen.generate_count_from_percentage(self.population_size,
                                                                                                min_percentage,
                                                                                                max_percentage)

            the_bests_to_select.append(the_best_to_possible_select)

        return f_table, cr_table, the_bests_to_select

    def next_epoch(self):
        """
        Perform the next epoch of the SHADE algorithm.
        """
        f_table, cr_table, the_bests_to_select = self._initialize_parameters_for_epoch()

        mutant = self._mutate(self._pop, the_bests_to_select, f_table)

        fix_boundary_constraints(mutant, self.boundary_constraints_fun)

        # Crossover step
        trial = self._binomial_crossing.crossover_population(self._pop, mutant, cr_table)

        # Evaluate fitness values for the trial population
        trial.update_fitness_values(self._function.eval, self.parallel_processing)

        # Selection step
        new_pop = self._selection(self._pop, trial, f_table, cr_table)

        # Archive management
        self._archive = self._archive_reduction.reduce_archive(self._archive, self._archive_size, self.population_size)

        # Update the population
        self._pop = new_pop

        # Update the memory for CR and F
        self._update_memory(self._successF, self._successCr, self._difference_fitness_success)
