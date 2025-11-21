import copy
import random
from typing import List

import numpy as np
from detpy.DETAlgs.base import BaseAlg
from detpy.DETAlgs.data.alg_data import SPSLShadeEIGDATA
from detpy.DETAlgs.methods.methods_sps_lshade_eig import archive_reduction, calculate_best_member_count, \
    mutation_internal, binomial_crossover_internal
from detpy.models.enums.boundary_constrain import fix_boundary_constraints_with_parent

from detpy.models.enums.optimization import OptimizationType
from detpy.models.member import Member
from detpy.models.population import Population


class SPS_LSHADE_EIG(BaseAlg):
    def __init__(self, params: SPSLShadeEIGDATA, db_conn=None, db_auto_write=False):
        super().__init__(SPS_LSHADE_EIG.__name__, params, db_conn, db_auto_write)

        self._h = params.memory_size
        self._memory_F = np.full(self._h, params.f_init)
        self._memory_Cr = np.full(self._h, params.cr_init)
        self._er_init = params.er_init
        self._population_size_reduction_strategy = params.population_reduction_strategy
        self._learning_rate_init = params.learning_rate_init
        self._learning_rate = self._learning_rate_init
        self._cr_min = params.cr_min
        self._cr_max = params.cr_max
        self._p_best_fraction = params.p_best_fraction
        self._successCr = []
        self._successF = []
        self._difference_fitness_success = []

        self._archive_size_sps = self.population_size
        self._archive_sps = list(copy.deepcopy(self._pop.members))
        self._archive = []

        self._w_ext = params.w_ext

        self._memory_Er = np.full(self._h, params.er_init)
        self._success_history_idx = 0
        self._successEr = []

        self._q = params.q

        self._min_pop_size = params.minimum_population_size
        self._recently_consecutive_unsuccessful_updates_table = [0] * self.population_size

        self._start_population_size = self.population_size

        self._cov_matrix = np.eye(self._pop.arg_num)
        self._learning_rate = self._learning_rate_init

        self._w_er = params.w_er
        self._w_f = params.w_f
        self._w_cr = params.w_cr

    def _update_covariance_matrix(self):
        """Update the covariance matrix based on the current population."""
        population_matrix = np.array([
            [chromosome.real_value for chromosome in member.chromosomes]
            for member in self._pop.members
        ])
        new_cov = np.cov(population_matrix, rowvar=False)
        self._learning_rate = self._learning_rate_init * (1 - (self.nfe) / self.nfe_max)
        self._learning_rate = max(self._learning_rate, 1e-5)
        self._cov_matrix = (1 - self._learning_rate) * self._cov_matrix + self._learning_rate * new_cov

    def _mutate(self, population: Population, the_best_to_select_table: List[int], f_table: List[float]) -> Population:
        """
        Perform mutation step for the population in SPS-L-SHADE-EIG.

        Parameters:
        - population (Population): The population to mutate.
        - the_best_to_select_table (List[int]): List of the number of the best members to select.
        - f_table (List[float]): List of scaling factors for mutation.

        Returns: A new Population with mutated members.
        """
        new_members = []
        sum_archive_and_population = np.concatenate((population.members, self._archive))

        for i in range(population.size):
            # Exclude the current index `i` for r1
            r1_candidates = [idx for idx in range(len(population.members)) if idx != i]
            r1 = np.random.choice(r1_candidates, 1, replace=False)[0]

            # Exclude `i` and `r1` for r2
            r2_candidates = [idx for idx in range(len(sum_archive_and_population)) if idx != i and idx != r1]
            r2 = np.random.choice(r2_candidates, 1, replace=False)[0]

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

    def _add_to_sps_archive(self, member: Member):
        """
        Add a member to the SPS archive, maintaining its size limit.

        Parameters:
        - member (Member): The member to be added to the SPS archive.
                           It is an instance of the `Member` class containing chromosomes.
        """
        if len(self._archive_sps) >= self._archive_size_sps:
            self._archive_sps.pop(0)
        self._archive_sps.append(member)

    def _selection(self, origin_population: Population, modified_population: Population, ftable: List[float],
                   cr_table: List[float], er_table: List[float]) -> Population:
        """
        Perform selection operation for the population.

        Parameters:
        - origin_population (Population): The original population.
        - modified_population (Population): The modified population
        - ftable (List[float]): List of scaling factors for mutation.
        - cr_table (List[float]): List of crossover rates.
        - er_table (List[float]): List of eigenvector crossover probabilities (Er),
          determining how likely each individual uses EIG-based crossover instead of
          the standard binomial crossover.

        Returns: A new population with the selected members.
        """
        new_members = []
        for i in range(origin_population.size):

            if origin_population.optimization == OptimizationType.MINIMIZATION:
                better = origin_population.members[i].fitness_value > modified_population.members[i].fitness_value
            else:
                better = origin_population.members[i].fitness_value < modified_population.members[i].fitness_value

            if better:
                self._archive.append(copy.deepcopy(origin_population.members[i]))
                self._add_to_sps_archive(copy.deepcopy(modified_population.members[i]))
                self._successF.append(ftable[i])
                self._successCr.append(cr_table[i])
                self._successEr.append(er_table[i])
                self._difference_fitness_success.append(
                    abs(origin_population.members[i].fitness_value - modified_population.members[i].fitness_value)
                )
                new_members.append(copy.deepcopy(modified_population.members[i]))
                self._recently_consecutive_unsuccessful_updates_table[i] = 0
            else:
                new_members.append(copy.deepcopy(origin_population.members[i]))
                self._recently_consecutive_unsuccessful_updates_table[i] += 1

            if self._recently_consecutive_unsuccessful_updates_table[i] >= self._q and self._archive_sps:
                new_members[i] = random.choice(self._archive_sps)
                self._recently_consecutive_unsuccessful_updates_table[i] = 0

        new_population = copy.deepcopy(origin_population)
        new_population.members = np.array(new_members)

        # Elitism: ensure the best member from the previous population is retained
        if origin_population.optimization == OptimizationType.MINIMIZATION:
            prev_best = min(origin_population.members, key=lambda m: m.fitness_value)
            curr_best = min(new_population.members, key=lambda m: m.fitness_value)
            if prev_best.fitness_value < curr_best.fitness_value:
                worst_idx = max(range(len(new_population.members)),
                                key=lambda i: new_population.members[i].fitness_value)
                new_population.members[worst_idx] = copy.deepcopy(prev_best)
        else:
            prev_best = max(origin_population.members, key=lambda m: m.fitness_value)
            curr_best = max(new_population.members, key=lambda m: m.fitness_value)
            if prev_best.fitness_value > curr_best.fitness_value:
                worst_idx = min(range(len(new_population.members)),
                                key=lambda i: new_population.members[i].fitness_value)
                new_population.members[worst_idx] = copy.deepcopy(prev_best)

        return new_population

    def _initialize_parameters_for_epoch(self):
        """
        Initialize the parameters for the next epoch of the SPS-L-SHADE-EIG algorithm.
        f_table: List of scaling factors for mutation.
        cr_table: List of crossover rates.
        er_table: List of eigenvector crossover probabilities (Er).
        the_bests_to_select: List of the number of the best members to select because in crossover we need to select
        the best members from the population for one factor.

        Returns:
        - f_table (List[float]): List of scaling factors for mutation.
        - cr_table (List[float]): List of crossover rates.
        - er_table (List[float]): List of eigenvector crossover probabilities (Er).
        - the_bests_to_select (List[int]): List of the number of the best members to possibly select.
        """
        f_table, cr_table, er_table, the_bests_to_select = [], [], [], []
        for i in range(self._pop.size):

            ri = np.random.randint(0, self._h)
            mean_f = self._memory_F[ri]
            mean_cr = self._memory_Cr[ri]

            cr = np.clip(np.random.normal(mean_cr, self._w_cr), self._cr_min, self._cr_max)

            while True:
                f = np.random.standard_cauchy() * self._w_f + mean_f
                if f > 0:
                    break
            f = min(f, 1.0)
            mean_er = self._memory_Er[ri]
            er = np.clip(np.random.normal(mean_er, self._w_er), 0.0, 1.0)

            f_table.append(f)
            cr_table.append(cr)
            er_table.append(er)

            bests_to_select = calculate_best_member_count(self.population_size, self._p_best_fraction)
            the_bests_to_select.append(bests_to_select)

        return f_table, cr_table, er_table, the_bests_to_select

    def _eig_crossover_internal(self, x: Member, v: Member, cr: float) -> Member:
        """
        Perform eigenvector-based crossover between two members x and v.

        Parameters:
        - x (Member) : The original member (parent) from the population.
             It is an instance of the `Member` class containing chromosomes.
        - v (Member) : The mutated member (donor vector) generated during the mutation step.
             It is also an instance of the `Member` class containing chromosomes.
        - cr (float) : The crossover rate (a float between 0 and 1) that determines the probability
              of inheriting a gene from the donor vector `v`.

        Returns:
        - Member: A new member (child) created by combining genes from `x` and `v`
                  based on the crossover rate `cr` and the eigenvector transformation.
                  The child is an instance of the `Member` class.
        """
        x_values = np.array([chromosome.real_value for chromosome in x.chromosomes])
        v_values = np.array([chromosome.real_value for chromosome in v.chromosomes])

        Q = np.linalg.eigh(self._cov_matrix)[1]
        xt, vt = Q.T @ x_values, Q.T @ v_values

        randmask = (np.random.rand(len(x_values)) < cr)
        ut = np.where(randmask, vt, xt)

        jrand = np.random.randint(0, len(ut))
        ut[jrand] = vt[jrand]
        u_values = Q @ ut
        new_member = copy.deepcopy(x)
        for i, chromosome in enumerate(new_member.chromosomes):
            chromosome.real_value = u_values[i]
        return new_member

    def _crossing_with_er(self, origin_population: Population, mutated_population: Population,
                          cr_table: List[float], er_table: List[float]) -> Population:
        """
        Perform crossover operation with eigenvector-based crossover probability (Er).

        Parameters:
        - origin_population (Population): The original population of members.
        - mutated_population (Population): The mutated population of members.
        - cr_table (List[float]): List of crossover rates for each member.
        - er_table (List[float]): List of eigenvector crossover probabilities (Er) for each member.

        Returns:
            Population: A new population created by combining genes from the original and mutated populations
                        based on the crossover rates and eigenvector probabilities.
        """
        new_members = []
        for i in range(origin_population.size):
            if np.random.rand() < er_table[i]:
                member = self._eig_crossover_internal(origin_population.members[i], mutated_population.members[i],
                                                      cr_table[i])
            else:
                member = binomial_crossover_internal(origin_population.members[i], mutated_population.members[i],
                                                     cr_table[i])
            new_members.append(member)
        new_population = copy.deepcopy(origin_population)
        new_population.members = np.array(new_members)
        return new_population

    def _update_population_size(self, nfe: int, total_nfe: int, start_pop_size: int, min_pop_size: int):
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
        self._archive_size_sps = new_size

    def _update_memory(self):
        """
        Update the memory of successful parameters (F, Cr, Er) based on the successes in the current generation.
        """
        if len(self._successF) == 0:
            return

        weights = np.array(self._difference_fitness_success)
        weights /= np.sum(weights)

        # Lehmer mean for F
        mean_F = np.sum(weights * np.square(self._successF)) / np.sum(weights * self._successF)

        # Weighted mean for Cr and Er
        mean_Cr = np.sum(weights * np.array(self._successCr))
        mean_Er = np.sum(weights * np.array(self._successEr))

        self._memory_Cr[self._success_history_idx] = mean_Cr
        self._memory_F[self._success_history_idx] = mean_F
        self._memory_Er[self._success_history_idx] = mean_Er

        self._success_history_idx = (self._success_history_idx + 1) % self._h

        self._successF.clear()
        self._successCr.clear()
        self._successEr.clear()
        self._difference_fitness_success.clear()

    def next_epoch(self):
        """
        Perform the next epoch of the SPS-L-SHADE-EIG algorithm.
        """
        f_table, cr_table, er_table, the_bests_to_select = self._initialize_parameters_for_epoch()
        mutant = self._mutate(self._pop, the_bests_to_select, f_table)
        trial = self._crossing_with_er(self._pop, mutant, cr_table, er_table)

        fix_boundary_constraints_with_parent(self._pop, trial, self.boundary_constraints_fun)

        trial.update_fitness_values(self._function.eval, self.parallel_processing)

        self._pop = self._selection(self._pop, trial, f_table, cr_table, er_table)

        self._archive = archive_reduction(self._archive, self.population_size, self._w_ext)
        self._update_covariance_matrix()

        self._update_population_size(self.nfe, self.nfe_max, self._start_population_size,
                                     self._min_pop_size)

        self._update_memory()
