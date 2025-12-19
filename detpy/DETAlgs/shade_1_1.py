import copy
from typing import List

import numpy as np

from detpy.DETAlgs.archive_reduction.archive_reduction import ArchiveReduction
from detpy.DETAlgs.base import BaseAlg
from detpy.DETAlgs.crossover_methods.binomial_crossover import BinomialCrossover
from detpy.DETAlgs.data.alg_data import Shade_1_1_Data
from detpy.DETAlgs.mutation_methods.current_to_pbest_1 import MutationCurrentToPBest1
from detpy.DETAlgs.random.index_generator import IndexGenerator
from detpy.DETAlgs.random.random_value_generator import RandomValueGenerator

from detpy.models.enums.boundary_constrain import fix_boundary_constraints_with_parent
from detpy.models.enums.optimization import OptimizationType

from detpy.models.population import Population


class SHADE_1_1(BaseAlg):
    """
        SHADE 1.1: Success-History based Adaptive Differential Evolution

        References:
        Ryoji Tanabe and Alex Fukunaga Graduate School of Arts and Sciences The University of Tokyo
    """

    def __init__(self, params: Shade_1_1_Data, db_conn=None, db_auto_write=False):
        super().__init__(SHADE_1_1.__name__, params, db_conn, db_auto_write)

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

        # Define a terminal value for memory_Cr. It indicates that no successful Cr was found in the epoch.
        self._TERMINAL = np.nan

        # We need this value for checking close to zero in update_memory
        self._EPSILON = 0.00001

        self._index_gen = IndexGenerator()
        self._random_value_gen = RandomValueGenerator()
        self._binomial_crossing = BinomialCrossover()
        self._archive_reduction = ArchiveReduction()

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
            selected_best_member = best_members[np.random.randint(0, len(best_members))]

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

    def selection(self, origin_population: Population, modified_population: Population, ftable: List[float],
                  cr_table: List[float]):
        """
        Perform selection operation for the population.

        Parameters:
        - origin_population (Population): The original population.
        - modified_population (Population): The modified population
        - ftable (List[float]): List of scaling factors for mutation.
        - cr_table (List[float]): List of crossover rates.

        Returns: A new population with the selected members.
        """
        optimization = origin_population.optimization
        new_members = []
        for i in range(origin_population.size):
            if optimization == OptimizationType.MINIMIZATION:
                if origin_population.members[i] <= modified_population.members[i]:
                    new_members.append(copy.deepcopy(origin_population.members[i]))
                else:
                    self._archive.append(copy.deepcopy(origin_population.members[i]))
                    self._successF.append(ftable[i])
                    self._successCr.append(cr_table[i])
                    self._difference_fitness_success.append(
                        origin_population.members[i].fitness_value - modified_population.members[i].fitness_value)
                    new_members.append(copy.deepcopy(modified_population.members[i]))
            elif optimization == OptimizationType.MAXIMIZATION:
                if origin_population.members[i] >= modified_population.members[i]:
                    new_members.append(copy.deepcopy(origin_population.members[i]))
                else:
                    self._archive.append(copy.deepcopy(origin_population.members[i]))
                    self._successF.append(ftable[i])
                    self._successCr.append(cr_table[i])
                    self._difference_fitness_success.append(
                        modified_population.members[i].fitness_value - origin_population.members[i].fitness_value)
                    new_members.append(copy.deepcopy(modified_population.members[i]))

        new_population = Population.with_new_members(origin_population, new_members)
        return new_population

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
                cr_new = np.sum(weights * success_cr * success_cr) / np.sum(weights * success_cr)
                cr_new = np.clip(cr_new, 0, 1)
                self._memory_Cr[self._k_index] = cr_new

            f_new = np.sum(weights * success_f * success_f) / np.sum(weights * success_f)
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


            f = self._random_value_gen.generate_cauchy_greater_than_zero(self._memory_F[ri], 0.1, 0.0, 1.0)

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
        f_table, cr_table, the_bests_to_select = self.initialize_parameters_for_epoch()

        mutant = self.mutate(self._pop, the_bests_to_select, f_table)

        # Crossover step
        trial = self._binomial_crossing.crossover_population(self._pop, mutant, cr_table)

        fix_boundary_constraints_with_parent(self._pop, trial, self.boundary_constraints_fun)

        # Evaluate fitness values for the trial population
        trial.update_fitness_values(self._function.eval, self.parallel_processing)

        # Selection step
        new_pop = self.selection(self._pop, trial, f_table, cr_table)

        # Archive management
        self._archive = self._archive_reduction.reduce_archive(self._archive, self._archive_size, self.population_size)

        # Update the population
        self._pop = new_pop

        # Update the memory for CR and F
        self.update_memory(self._successF, self._successCr, self._difference_fitness_success)
