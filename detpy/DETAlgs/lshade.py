import copy
from typing import List

import numpy as np
from detpy.DETAlgs.base import BaseAlg
from detpy.DETAlgs.data.alg_data import LShadeData
from detpy.DETAlgs.methods.methods_lshade import calculate_best_member_count, crossing, \
    archive_reduction
from detpy.DETAlgs.mutation_methods.current_to_pbest_1 import MutationCurrentToPBest1

from detpy.models.enums.boundary_constrain import fix_boundary_constraints
from detpy.models.enums.optimization import OptimizationType

from detpy.models.population import Population


class LSHADE(BaseAlg):
    """
        LSHADE: Improving the Search Performance of SHADE Using Linear Population Size Reduction

        References:
        Ryoji Tanabe and Alex Fukunaga Graduate School of Arts and Sciences The University of Tokyo
    """

    def __init__(self, params: LShadeData, db_conn=None, db_auto_write=False):
        super().__init__(LSHADE.__name__, params, db_conn, db_auto_write)

        self.H = params.memory_size  # Memory size for f and cr adaptation
        self.memory_F = np.full(self.H, 0.5)  # Initial memory for F
        self.memory_Cr = np.full(self.H, 0.5)  # Initial memory for Cr

        self.successCr = []
        self.successF = []
        self.difference_fitness_success = []

        self.min_the_best_percentage = 2 / self.population_size  # Minimal percentage of the best members to consider

        self.archive_size = self.population_size  # Size of the archive
        self.archive = []  # Archive for storing the members from old populations

        self.min_pop_size = params.minimum_population_size  # Minimal population size

        self.start_population_size = self.population_size

        self.population_size_reduction_strategy = params.population_reduction_strategy

    def update_population_size(self, nfe: int, total_nfe: int, start_pop_size: int, min_pop_size: int):
        """
        Calculate new population size using Linear Population Size Reduction (LPSR).

        Parameters:
        - start_pop_size (int): The initial population size.
        - nfe (int): The current function evaluations.
        - total_nfe (int): The total number of function evaluations.
        - min_pop_size (int): The minimum population size.
        """
        new_size = self.population_size_reduction_strategy.get_new_population_size(
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

        sum_archive_and_population = np.concatenate((population.members, self.archive))

        for i in range(population.size):
            r1 = np.random.choice(len(population.members), 1, replace=False)[0]

            # Archive is included population and archive members
            r2 = np.random.choice(len(sum_archive_and_population), 1, replace=False)[0]

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
        new_population = Population(
            lb=population.lb,
            ub=population.ub,
            arg_num=population.arg_num,
            size=population.size,
            optimization=population.optimization
        )
        new_population.members = np.array(new_members)
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
                    self.archive.append(copy.deepcopy(origin_population.members[i]))
                    self.successF.append(ftable[i])
                    self.successCr.append(cr_table[i])
                    self.difference_fitness_success.append(
                        origin_population.members[i].fitness_value - modified_population.members[i].fitness_value)
                    new_members.append(copy.deepcopy(modified_population.members[i]))
            elif optimization == OptimizationType.MAXIMIZATION:
                if origin_population.members[i] >= modified_population.members[i]:
                    new_members.append(copy.deepcopy(origin_population.members[i]))
                else:
                    self.archive.append(copy.deepcopy(origin_population.members[i]))
                    self.successF.append(ftable[i])
                    self.successCr.append(cr_table[i])
                    self.difference_fitness_success.append(
                        modified_population.members[i].fitness_value - origin_population.members[i].fitness_value)
                    new_members.append(copy.deepcopy(modified_population.members[i]))

        new_population = Population(
            lb=origin_population.lb,
            ub=origin_population.ub,
            arg_num=origin_population.arg_num,
            size=origin_population.size,
            optimization=origin_population.optimization
        )
        new_population.members = np.array(new_members)
        return new_population

    def update_memory(self, success_f: List[float], success_cf: List[float], difference_fitness_success: List[float]):
        """
        Update the memory for the crossover rates and scaling factors based on the success of the trial vectors.

        Parameters:
        - success_f (List[float]): List of scaling factors that led to better trial vectors.
        - success_cf (List[float]): List of crossover rates that led to better trial vectors.
        - difference_fitness_success (List[float]): List of differences in objective function values (|f(u_k, G) - f(x_k, G)|).
        """
        if len(success_f) > 0:
            total = np.sum(difference_fitness_success)
            if total == 0:
                return

            weights = difference_fitness_success / total
            f_new = np.sum(weights * success_f * success_f) / np.sum(weights * success_f)
            f_new = np.clip(f_new, 0, 1)
            cr_new = np.sum(weights * success_cf)
            cr_new = np.clip(cr_new, 0, 1)

            self.memory_F = np.append(self.memory_F[1:], f_new)
            self.memory_Cr = np.append(self.memory_Cr[1:], cr_new)

            # Reset the lists for the next generation
            self.successF = []
            self.successCr = []
            self.difference_fitness_success = []

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
            ri = np.random.randint(0, self.H)
            mean_f = self.memory_F[ri]
            mean_cr = self.memory_Cr[ri]

            cr = np.random.normal(mean_cr, 0.1)
            cr = np.clip(cr, 0.0, 1.0)

            while True:
                f = np.random.standard_cauchy() * 0.1 + mean_f
                if f > 0:
                    break

            f = min(f, 1.0)

            f_table.append(f)
            cr_table.append(cr)

            the_best_to_possible_select = calculate_best_member_count(
                self.population_size)

            the_bests_to_select.append(the_best_to_possible_select)

        return f_table, cr_table, the_bests_to_select

    def next_epoch(self):
        """
        Perform the next epoch of the SHADE algorithm.
        """
        f_table, cr_table, the_bests_to_select = self.initialize_parameters_for_epoch()

        mutant = self.mutate(self._pop, the_bests_to_select, f_table)

        fix_boundary_constraints(mutant, self.boundary_constraints_fun)

        # Crossover step
        trial = crossing(self._pop, mutant, cr_table)

        # Evaluate fitness values for the trial population
        trial.update_fitness_values(self._function.eval, self.parallel_processing)

        # Selection step
        new_pop = self.selection(self._pop, trial, f_table, cr_table)

        # Archive management
        self.archive = archive_reduction(self.archive, self.archive_size, self.population_size)

        # Update the population
        self._pop = new_pop

        # Update the memory for CR and F
        self.update_memory(self.successF, self.successCr, self.difference_fitness_success)

        self.update_population_size(self.nfe, self.nfe_max, self.start_population_size,
                                    self.min_pop_size)

