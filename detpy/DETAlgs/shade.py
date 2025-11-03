import copy
from typing import List

import numpy as np
from detpy.DETAlgs.base import BaseAlg
from detpy.DETAlgs.data.alg_data import ShadeData
from detpy.DETAlgs.methods.methods_shade import mutation_internal, crossing, archive_reduction, \
    calculate_best_member_count

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
            r1 = np.random.choice(len(population.members), 1, replace=False)[0]

            # Archive is included population and archive members
            r2 = np.random.choice(len(sum_archive_and_population), 1, replace=False)[0]

            # Select top p-best members from the population
            best_members = population.get_best_members(the_best_to_select_table[i])

            # Randomly select one of the p-best members
            selected_best_member = best_members[np.random.randint(0, len(best_members))]

            # Apply the mutation formula (current-to-pbest strategy)
            mutated_member = mutation_internal(
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

    def _selection(self, origin_population: Population, modified_population: Population, ftable: List[float],
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

        new_population = Population(
            lb=origin_population.lb,
            ub=origin_population.ub,
            arg_num=origin_population.arg_num,
            size=origin_population.size,
            optimization=origin_population.optimization
        )
        new_population.members = np.array(new_members)
        return new_population

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
            f_new = np.sum(weights * success_f * success_f) / np.sum(weights * success_f)
            f_new = np.clip(f_new, 0, 1)
            cr_new = np.sum(weights * success_cr)
            cr_new = np.clip(cr_new, 0, 1)

            # self.memory_F = np.append(self.memory_F[1:], f_new)
            # self.memory_Cr = np.append(self.memory_Cr[1:], cr_new)

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
            ri = np.random.randint(0, self._H)
            mean_f = self._memory_F[ri]
            mean_cr = self._memory_Cr[ri]

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
        f_table, cr_table, the_bests_to_select = self._initialize_parameters_for_epoch()

        mutant = self._mutate(self._pop, the_bests_to_select, f_table)

        fix_boundary_constraints(mutant, self.boundary_constraints_fun)

        # Crossover step
        trial = crossing(self._pop, mutant, cr_table)

        # Evaluate fitness values for the trial population
        trial.update_fitness_values(self._function.eval, self.parallel_processing)

        # Selection step
        new_pop = self._selection(self._pop, trial, f_table, cr_table)

        # Archive management
        self._archive = archive_reduction(self._archive, self._archive_size, self.population_size)

        # Update the population
        self._pop = new_pop

        # Update the memory for CR and F
        self._update_memory(self._successF, self._successCr, self._difference_fitness_success)
