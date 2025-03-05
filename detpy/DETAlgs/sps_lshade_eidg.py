import copy
import random
from typing import List

import numpy as np
from detpy.DETAlgs.base import BaseAlg
from detpy.DETAlgs.data.alg_data import SPSLShadeEEIDGDATA
from detpy.DETAlgs.methods.methods_lshade import mutation_internal, calculate_best_member_count, archive_reduction

from detpy.models.enums.boundary_constrain import fix_boundary_constraints
from detpy.models.enums.optimization import OptimizationType
from detpy.models.population import Population


class SPS_LSHADE_EIDG(BaseAlg):
    def __init__(self, params: SPSLShadeEEIDGDATA, db_conn=None, db_auto_write=False):
        super().__init__(SPS_LSHADE_EIDG.__name__, params, db_conn, db_auto_write)

        self.H = params.memory_size
        self.memory_F = np.full(self.H, 0.5)
        self.memory_Cr = np.full(self.H, 0.5)
        self.successCr = []
        self.successF = []
        self.difference_fitness_success = []

        self.archive_size_sps = self.population_size  # Archive for SPS framework
        self.archive_sps = []  # Stores recently successful parents
        self.archive = []  # Standard archive
        self.archive_size = self.population_size

        self.q = params.q  # Stagnation threshold
        self.nfe = 0  # Function evaluations counter
        self.min_pop_size = params.minimum_population_size
        self.recently_consecutive_unsuccessful_updates_table = [0] * self.population_size

        self.start_population_size = self.population_size
        self.nfe_max = self.calculate_max_evaluations_lpsr(self.population_size)
        self.cov_matrix = np.eye(self._pop.arg_num)  # Initial covariance matrix
        self.learning_rate = 0.3  # Adaptive learning rate for covariance matrix

    def update_covariance_matrix(self):
        population_matrix = np.array([
            [chromosome.real_value for chromosome in member.chromosomes]
            for member in self._pop.members
        ])
        new_cov = np.cov(population_matrix, rowvar=False)
        self.cov_matrix = (1 - self.learning_rate) * self.cov_matrix + self.learning_rate * new_cov
        self.learning_rate *= 0.99  # Gradually decrease learning rate

    def mutate(self, population: Population, the_best_to_select_table: List[int], f_table: List[float]) -> Population:
        new_members = []
        sum_archive_and_population = np.concatenate((population.members, self.archive))

        for i in range(population.size):
            r2 = np.random.randint(0, len(sum_archive_and_population))
            r1 = np.random.randint(0, population.size)
            best_members = population.get_best_members(the_best_to_select_table[i])
            selected_best_member = random.choice(best_members)
            mutated_member = mutation_internal(
                base_member=population.members[i],
                best_member=selected_best_member,
                r1=population.members[r1],
                r2=sum_archive_and_population[r2],
                f=f_table[i]
            )
            new_members.append(mutated_member)

        new_population = copy.deepcopy(population)
        new_population.members = np.array(new_members)
        return new_population

    def add_to_sps_archive(self, member):
        if len(self.archive_sps) >= self.archive_size_sps:
            self.archive_sps.pop(0)
        self.archive_sps.append(member)

    def selection(self, origin_population: Population, modified_population: Population, ftable: List[float],
                  cr_table: List[float]):
        new_members = []
        for i in range(origin_population.size):
            if origin_population.optimization == OptimizationType.MINIMIZATION:
                better = origin_population.members[i].fitness_value > modified_population.members[i].fitness_value
            else:
                better = origin_population.members[i].fitness_value < modified_population.members[i].fitness_value

            if better:
                self.archive.append(copy.deepcopy(origin_population.members[i]))
                self.add_to_sps_archive(copy.deepcopy(modified_population.members[i]))
                self.successF.append(ftable[i])
                self.successCr.append(cr_table[i])
                self.difference_fitness_success.append(
                    abs(origin_population.members[i].fitness_value - modified_population.members[i].fitness_value)
                )
                new_members.append(copy.deepcopy(modified_population.members[i]))
                self.recently_consecutive_unsuccessful_updates_table[i] = 0
            else:
                new_members.append(copy.deepcopy(origin_population.members[i]))
                self.recently_consecutive_unsuccessful_updates_table[i] += 1

            if self.recently_consecutive_unsuccessful_updates_table[i] >= self.q:
                new_members[i] = random.choice(self.archive_sps)
                self.recently_consecutive_unsuccessful_updates_table[i] = 0

        new_population = copy.deepcopy(origin_population)
        new_population.members = np.array(new_members)
        return new_population

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

    def eig_crossover_internal(self, x, v, CR):
        x_values = np.array([chromosome.real_value for chromosome in x.chromosomes])
        v_values = np.array([chromosome.real_value for chromosome in v.chromosomes])
        Q = np.linalg.eigh(self.cov_matrix)[1]
        xt, vt = Q.T @ x_values, Q.T @ v_values
        ut = np.where(np.random.rand(len(x_values)) < CR, vt, xt)
        ut[np.random.randint(0, len(ut))] = vt[np.random.randint(0, len(ut))]
        u_values = Q @ ut
        new_member = copy.deepcopy(x)
        for i, chromosome in enumerate(new_member.chromosomes):
            chromosome.real_value = u_values[i]
        return new_member

    def eig_crossing(self, origin_population, mutated_population, cr_table):
        new_members = [
            self.eig_crossover_internal(origin_population.members[i], mutated_population.members[i], cr_table[i])
            for i in range(origin_population.size)
        ]

        new_population = copy.deepcopy(origin_population)
        new_population.members = np.array(new_members)
        return new_population


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
        print("New size: ", new_size)
        self._pop.resize(new_size)

        # Update archive size proportionally
        self.archive_size_sps = new_size


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
    def next_epoch(self):
        f_table, cr_table, the_bests_to_select = self.initialize_parameters_for_epoch()
        mutant = self.mutate(self._pop, the_bests_to_select, f_table)
        fix_boundary_constraints(mutant, self.boundary_constraints_fun)
        trial = self.eig_crossing(self._pop, mutant, cr_table)
        trial.update_fitness_values(self._function.eval, self.parallel_processing)
        self._pop = self.selection(self._pop, trial, f_table, cr_table)
        self.archive = archive_reduction(self.archive, self.archive_size, self.population_size)
        self.update_covariance_matrix()
        self.update_population_size(self.start_population_size, self._epoch_number, self.num_of_epochs,
                                    self.min_pop_size)
        self._epoch_number += 1