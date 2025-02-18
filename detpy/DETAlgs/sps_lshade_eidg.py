import copy
import random
from typing import List

import numpy as np
from detpy.DETAlgs.base import BaseAlg
from detpy.DETAlgs.data.alg_data import LShadeData, SPSLShadeEEIDGDATA
from detpy.DETAlgs.methods.methods_lshade import mutation_internal, calculate_best_member_count, crossing, \
    archive_reduction

from detpy.models.enums.boundary_constrain import fix_boundary_constraints
from detpy.models.enums.optimization import OptimizationType

from detpy.models.population import Population


class SPS_LSHADE_EIDG(BaseAlg):
    """
        LSHADE: Improving the Search Performance of SHADE Using Linear Population Size Reduction

        References:
        Ryoji Tanabe and Alex Fukunaga Graduate School of Arts and Sciences The University of Tokyo
    """

    def __init__(self, params: SPSLShadeEEIDGDATA, db_conn=None, db_auto_write=False):
        super().__init__(SPS_LSHADE_EIDG.__name__, params, db_conn, db_auto_write)

        self.H = params.memory_size  # Memory size for f and cr adaptation
        self.memory_F = np.full(self.H, 0.5)  # Initial memory for F
        self.memory_Cr = np.full(self.H, 0.5)  # Initial memory for Cr

        self.successCr = []
        self.successF = []
        self.difference_fitness_success = []

        self.min_the_best_percentage = 2 / self.population_size  # Minimal percentage of the best members to consider

        self.archive_size_sps =  self.population_size  # Size of the archive
        self.archive_sps = copy.deepcopy(self._pop.members.tolist())  # Archive for storing the members from old populations

        self.archive = []
        self.archive_size = self.population_size  # Size of the archive

        self.q = params.q



        self.nfe = 0  # Number of function evaluations
        self.min_pop_size = params.minimum_population_size  # Minimal population size
        self.recently_consecutive_unsuccessful_updates_table = [0] * self.population_size

        self.start_population_size = self.population_size
        self.nfe_max = self.calculate_max_evaluations_lpsr(self.population_size)  # Max number of function evaluations

    def calculate_max_evaluations_lpsr(self, start_pop_size):
        """
        Calculate the maximum number of function evaluations for the Linear Population Size Reduction (LPSR) method.

        Parameters:
        - start_pop_size (int): The initial population size at the beginning of the evolution process.

        Returns: sum of the total number of evaluations for each generation.
        """
        total_evaluations = 0
        for generation in range(self.num_of_epochs):
            current_population_size = int(
                start_pop_size - (generation / self.num_of_epochs) * (start_pop_size - self.min_pop_size)
            )
            total_evaluations += current_population_size
        NFEmax = total_evaluations
        return NFEmax

    def update_population_size(self, start_pop_size: int, epoch: int, total_epochs: int, min_size: int):
        """
        Calculate new population size using Linear Population Size Reduction (LPSR).

        Parameters:
        - start_pop_size (int): The initial population size.
        - epoch (int): The current epoch.
        - total_epochs (int): The total number of epochs.
        - min_size (int): The minimum population size.
        """
        new_size = int(start_pop_size - (epoch / total_epochs) * (start_pop_size - min_size))
        self._pop.resize(new_size)

        # Update archive size proportionally
        self.archive_size_sps = new_size


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

        sum_archive_and_population = np.concatenate((population.members, self.archive_sps))

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

    def add_to_archive(self, member):
        """Dodaje nowe rozwiązanie do archiwum, usuwając najstarsze, jeśli limit został przekroczony."""
        if len(self.archive_sps) >= self.archive_size_sps:
            self.archive_sps.pop(0)  # Usuwa najstarszy element
        self.archive_sps.append(member)
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
                    self.recently_consecutive_unsuccessful_updates_table[i]+=1

                else:
                    self.archive.append(copy.deepcopy(origin_population.members[i]))
                    self.add_to_archive(copy.deepcopy(modified_population.members[i]))
                    self.successF.append(ftable[i])
                    self.successCr.append(cr_table[i])
                    self.difference_fitness_success.append(
                        origin_population.members[i].fitness_value - modified_population.members[i].fitness_value)
                    new_members.append(copy.deepcopy(modified_population.members[i]))

                if(self.recently_consecutive_unsuccessful_updates_table[i] >= self.q):
                    new_members.pop()
                    new_members.append(random.choice(self.archive_sps))
                    self.recently_consecutive_unsuccessful_updates_table[i] = 0

            elif optimization == OptimizationType.MAXIMIZATION:
                if origin_population.members[i] >= modified_population.members[i]:
                    new_members.append(copy.deepcopy(origin_population.members[i]))
                    self.recently_consecutive_unsuccessful_updates_table[i]+= 1
                else:
                    self.archive.append(copy.deepcopy(origin_population.members[i]))
                    self.add_to_archive(copy.deepcopy(origin_population.members[i]))
                    self.successF.append(ftable[i])
                    self.successCr.append(cr_table[i])
                    self.difference_fitness_success.append(
                        modified_population.members[i].fitness_value - origin_population.members[i].fitness_value)
                    new_members.append(copy.deepcopy(modified_population.members[i]))

                if(self.recently_consecutive_unsuccessful_updates_table[i] >= self.q):
                    new_members.pop()
                    new_members.append(random.choice(self.archive_sps))
                    self.recently_consecutive_unsuccessful_updates_table[i] = 0

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

        self.update_population_size(self.start_population_size, self._epoch_number, self.num_of_epochs,
                                    self.min_pop_size)

        self._epoch_number += 1

