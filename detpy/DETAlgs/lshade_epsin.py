import copy
import math

import numpy as np
from typing import List

from detpy.DETAlgs.base import BaseAlg
from detpy.DETAlgs.data.alg_data import LShadeEpsinData
from detpy.DETAlgs.methods.methods_lshade_epsin import calculate_best_member_count, crossing, archive_reduction
from detpy.DETAlgs.mutation_methods.current_to_pbest_1 import MutationCurrentToPBest1
from detpy.models.enums.boundary_constrain import fix_boundary_constraints_with_parent
from detpy.models.enums.optimization import OptimizationType
from detpy.models.population import Population


class LShadeEpsin(BaseAlg):
    def __init__(self, params: LShadeEpsinData, db_conn=None, db_auto_write=False):
        super().__init__(LShadeEpsin.__name__, params, db_conn, db_auto_write)

        self._H = params.memory_size
        self._memory_F = np.full(self._H, 0.5)
        self._memory_Cr = np.full(self._H, 0.5)

        self._success_cr = []
        self._success_f = []
        self._p = params.best_member_percentage
        self._success_freg = []
        self._difference_fitness_success = []
        self._difference_fitness_success_freg = []

        self._freq_memory = np.full(self._H, 0.5)

        self._archive = []

        self._archive_size = self.population_size

        self._archive.extend(copy.deepcopy(self._pop.members))

        self._min_pop_size = params.minimum_population_size
        self._start_population_size = self.population_size

        self._f_sin_freg = params.f_sin_freq

        self.population_size_reduction_strategy = params.population_reduction_strategy

        # Define a terminal value for memory_Cr. It indicates that no successful Cr was found in the epoch.
        self._TERMINAL = np.nan

        # We need this value for checking close to zero in update_memory
        self._EPSILON = 0.00001

        self._k_index = 0

    def _update_population_size(self, nfe: int, max_nfe: int, start_pop_size: int, min_pop_size: int):
        """
        Calculate new population size using Linear Population Size Reduction (LPSR).

        Parameters:
        - nfe (int): The current function evaluations.
        - max_nfe (int): The total number of function evaluations.
        - start_pop_size (int): The initial population size.
        - min_pop_size (int): The minimum population size.
        """
        new_size = self.population_size_reduction_strategy.get_new_population_size(
            nfe, max_nfe, start_pop_size, min_pop_size
        )

        self._pop.resize(new_size)

    def _mutate(self, population: Population, the_best_to_select_table: List[int], f_table: List[float]) -> Population:
        """
        Perform mutation step for the population in LSHADE-EpSin.

        Parameters:
        - population (Population): The population to mutate.
        - the_best_to_select_table (List[int]): List of the number of the best members to select.
        - f_table (List[float]): List of scaling factors for mutation.

        Returns: A new Population with mutated members.
        """
        new_members = []

        pa = np.concatenate((population.members, self._archive))

        for i in range(population.size):
            # Exclude the current index `i` for r1
            r1_candidates = [idx for idx in range(len(population.members)) if idx != i]
            r1 = np.random.choice(r1_candidates, 1, replace=False)[0]

            # Exclude `i` and `r1` for r2
            r2_candidates = [idx for idx in range(len(pa)) if idx != i and idx != r1]
            r2 = np.random.choice(r2_candidates, 1, replace=False)[0]

            best_members = population.get_best_members(the_best_to_select_table[i])
            best = best_members[np.random.randint(0, len(best_members))]

            mutant = MutationCurrentToPBest1.mutate(
                base_member=population.members[i],
                best_member=best,
                r1=population.members[r1],
                r2=pa[r2],
                f=f_table[i]
            )

            new_members.append(mutant)

        new_population = Population(
            lb=population.lb, ub=population.ub,
            arg_num=population.arg_num, size=population.size,
            optimization=population.optimization
        )
        new_population.members = np.array(new_members)

        return new_population

    def _selection(self, origin, modified, ftable, cr_table, freg_values):
        """
        Perform selection operation for the population.

        Parameters:
        - origin (Population): The original population.
        - modified (Population): The modified population
        - ftable (List[float]): List of scaling factors for mutation.
        - cr_table (List[float]): List of crossover rates.
        - freg_values (dict): Dictionary of frequency values for members.

        Returns: A new population with the selected members.
        """
        new_members = []

        for i in range(origin.size):
            if (origin.optimization == OptimizationType.MINIMIZATION and origin.members[i] <= modified.members[i]) or \
                    (origin.optimization == OptimizationType.MAXIMIZATION and origin.members[i] >= modified.members[i]):
                new_members.append(copy.deepcopy(origin.members[i]))
            else:
                self._archive.append(copy.deepcopy(origin.members[i]))
                self._success_f.append(ftable[i])
                self._success_cr.append(cr_table[i])
                df = abs(modified.members[i].fitness_value - origin.members[i].fitness_value)
                self._difference_fitness_success.append(df)

                if i in freg_values:
                    self._difference_fitness_success_freg.append(df)
                    self._success_freg.append(ftable[i])

                new_members.append(copy.deepcopy(modified.members[i]))

        new_population = Population(
            lb=origin.lb, ub=origin.ub,
            arg_num=origin.arg_num, size=origin.size,
            optimization=origin.optimization
        )
        new_population.members = np.array(new_members)
        return new_population

    def _reset_success_metrics(self):
        """Reset success metrics after memory update."""
        self._success_f = []
        self._success_cr = []
        self._success_freg = []
        self._difference_fitness_success = []
        self._difference_fitness_success_freg = []

    def _update_memory(self, success_f, success_cr, success_freg, df, df_freg):
        """
        Update the memory for the crossover rates, freg and scaling factors based on the success of the trial vectors.

        Parameters:
        - success_f (List[float]): List of scaling factors that led to better trial vectors.
        - success_cr (List[float]): List of crossover rates that led to better trial vectors.
        - success_freg (List[float]): List of freg values that led to better trial vectors.
        - df (List[float]): List of differences in objective function values (first part of evaluations) (|f(u_k, G) - f(x_k, G)|).
        - df_freg (List[float]): List of differences in objective function values for freg (second part of evaluations) (|f(u_k, G) - f(x_k, G)|).
        """

        is_generation_lower_than_half_population = self.nfe < (self.nfe_max / 2)

        if is_generation_lower_than_half_population:
            # frequency update only in the first half of the generations
            if len(success_freg) > 0:
                total = np.sum(df_freg)
                weights = df_freg / total
                denominator = np.sum(weights * success_freg)
                if denominator > 0:
                    freg_new = np.sum(weights * success_freg * success_freg) / np.sum(weights * success_freg)
                    freg_new = np.clip(freg_new, 0, 1)
                    random_index = np.random.randint(0, self._H)
                    self._freq_memory[random_index] = freg_new
            if len(success_f) > 0 and len(success_cr) > 0:
                total = np.sum(df)
                weights = df / total

                if np.isclose(total, 0.0, atol=self._EPSILON):
                    self._memory_Cr[self._k_index] = self._TERMINAL

                else:
                    cr_new = np.sum(weights * success_cr * success_cr) / np.sum(weights * success_cr)
                    cr_new = np.clip(cr_new, 0, 1)
                    self._memory_Cr[self._k_index] = cr_new

        else:
            if len(success_f) > 0 and len(success_cr) > 0:
                total = np.sum(df)
                weights = df / total

                if np.isclose(total, 0.0, atol=self._EPSILON):
                    self._memory_Cr[self._k_index] = self._TERMINAL

                else:
                    cr_new = np.sum(weights * success_cr * success_cr) / np.sum(weights * success_cr)
                    cr_new = np.clip(cr_new, 0, 1)
                    self._memory_Cr[self._k_index] = cr_new

                f_new = np.sum(weights * success_f * success_f) / np.sum(weights * success_f)
                f_new = np.clip(f_new, 0, 1)

                self._memory_F[self._k_index] = f_new

        self._k_index = (self._k_index + 1) % self._H
        self._reset_success_metrics()

    def _initialize_parameters_for_epoch(self):
        """
        Initialize the parameters for the next epoch of the LSHADE-EpSin algorithm.
         Parameters:
         f_table: List of scaling factors for mutation.
         cr_table: List of crossover rates.
         the_bests_to_select: List of the number of the best members to select because in crossover we need to select
         freg_values: Dictionary of frequency values for members.
         the best members from the population for one factor.

         Returns:
         - f_table (List[float]): List of scaling factors for mutation.
         - cr_table (List[float]): List of crossover rates.
         - bests_to_select (List[int]): List of the number of the best members to possibly select.
         - freg_values (dict): Dictionary of frequency values for members.
         """
        f_table = []
        cr_table = []
        bests_to_select = []
        freg_values = {}
        for idx in range(self._pop.size):
            ri = np.random.randint(0, self._H)

            if np.isnan(self._memory_Cr[ri]):
                cr = 0.0
            else:
                cr = np.random.normal(self._memory_Cr[ri], 0.1)
                cr = np.clip(cr, 0.0, 1.0)

            is_lower_then_half_populations = self.nfe < (self.nfe_max / 2)

            if is_lower_then_half_populations:
                is_not_adaptive_sinusoidal_increasing = np.random.rand() < 0.5
                if is_not_adaptive_sinusoidal_increasing:
                    f = 0.5 * (math.sin(2 * math.pi * self._f_sin_freg * self.nfe + math.pi) * (
                            self.nfe_max - self.nfe) / self.nfe_max + 1.0)
                else:

                    ri = np.random.randint(0, self._H)
                    elem_freg_external_memory = self._freq_memory[ri]

                    elem_cauchy = np.random.standard_cauchy() * 0.1 + elem_freg_external_memory
                    elem_cauchy = np.clip(elem_cauchy, 0.0, 1.0)  # nwm czy to potrzebne

                    f = 0.5 * math.sin(2 * math.pi * elem_cauchy * self.nfe) * (
                            self.nfe / self.nfe_max) + 1

                    freg_values[idx] = f
            else:
                while True:
                    f = np.random.standard_cauchy() * 0.1 + self._memory_F[ri]
                    if f > 0:
                        break

                if (f > 1):
                    f = 0.0

            f_table.append(f)
            cr_table.append(cr)
            bests_to_select.append(calculate_best_member_count(self.population_size, self._p))

        return f_table, cr_table, bests_to_select, freg_values

    def _perform_local_search(self):
        """
        Perform the Gaussian Walk local search phase as defined in the LSHADE-EpSin algorithm

        This procedure is executed once when the population size becomes very small
        (typically ≤ 20 individuals) in the later stage of evolution. It generates a small set
        of candidate solutions and refines them around the current best individual using a
        Gaussian-based local search mechanism.
        This approach allows the algorithm to exploit the best-found solution more precisely
        by performing small stochastic moves toward the global optimum, enhancing convergence
        when diversity is low.
        """

        num_candidates = 10
        G_ls = 250

        candidates = Population(
            lb=self._pop.lb, ub=self._pop.ub,
            arg_num=self._pop.arg_num, size=num_candidates,
            optimization=self._pop.optimization
        )
        candidates.generate_population()
        candidates.update_fitness_values(self._function.eval, self.parallel_processing)

        best_individual = min(candidates.members, key=lambda x: x.fitness_value) \
            if self._pop.optimization == OptimizationType.MINIMIZATION \
            else max(candidates.members, key=lambda x: x.fitness_value)

        for gen in range(G_ls):
            new_candidates = []
            for i, candidate in enumerate(candidates.members):
                x = [chromosome.real_value for chromosome in candidate.chromosomes]
                x_best = [chromosome.real_value for chromosome in best_individual.chromosomes]
                G = gen + 1

                sigma = np.abs((np.log(G) / G) * (np.array(x_best) - np.array(x)))

                epsilon = np.random.rand(self._pop.arg_num)
                epsilon_hat = np.random.rand(self._pop.arg_num)

                x_new = np.array(x_best) + epsilon * (np.array(x) - np.array(x_best)) + epsilon_hat * sigma
                x_new = np.clip(x_new, self._pop.lb, self._pop.ub)

                new_candidate = copy.deepcopy(candidate)
                for j, value in enumerate(x_new):
                    new_candidate.chromosomes[j].real_value = float(value)
                new_candidates.append(new_candidate)

            candidates.update_fitness_values(self._function.eval, self.parallel_processing)

            for i, candidate in enumerate(new_candidates):
                new_candidate = new_candidates[i]
                if (
                        self._pop.optimization == OptimizationType.MINIMIZATION and new_candidate.fitness_value < candidate.fitness_value) or \
                        (
                                self._pop.optimization == OptimizationType.MAXIMIZATION and new_candidate.fitness_value > candidate.fitness_value):
                    candidates.members[i] = new_candidate

        candidates.update_fitness_values(self._function.eval, self.parallel_processing)

        worst_indices = sorted(range(len(self._pop.members)), key=lambda i: self._pop.members[i].fitness_value,
                               reverse=(self._pop.optimization == OptimizationType.MINIMIZATION))[:num_candidates]

        for i, idx in enumerate(worst_indices):
            if (self._pop.optimization == OptimizationType.MINIMIZATION and
                candidates.members[i].fitness_value < self._pop.members[idx].fitness_value) or \
                    (self._pop.optimization == OptimizationType.MAXIMIZATION and
                     candidates.members[i].fitness_value > self._pop.members[idx].fitness_value):
                self._pop.members[idx] = candidates.members[i]

    def next_epoch(self):
        """
        Perform the next epoch of the LSHADE-EpSin algorithm.
        """
        f_table, cr_table, bests_to_select, freg_values = self._initialize_parameters_for_epoch()
        mutant = self._mutate(self._pop, bests_to_select, f_table)

        trial = crossing(self._pop, mutant, cr_table)
        fix_boundary_constraints_with_parent(self._pop, trial, self.boundary_constraints_fun)
        trial.update_fitness_values(self._function.eval, self.parallel_processing)

        self._pop = self._selection(self._pop, trial, f_table, cr_table, freg_values)
        self._archive = archive_reduction(self._archive, self._archive_size, self.population_size)
        self._update_memory(self._success_f, self._success_cr, self._success_freg, self._difference_fitness_success,
                            self._difference_fitness_success_freg)

        self._update_population_size(self.nfe, self.nfe_max, self._start_population_size,
                                     self._min_pop_size)

        if self._pop.size <= 20 and not hasattr(self, "_local_search_done"):
            self._local_search_done = True
            self._perform_local_search()
