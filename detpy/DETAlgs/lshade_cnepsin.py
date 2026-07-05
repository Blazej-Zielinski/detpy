import copy
import math
from typing import Dict, List

import numpy as np

from detpy.DETAlgs.archive_reduction.archive_reduction import ArchiveReduction
from detpy.DETAlgs.base import BaseAlg
from detpy.DETAlgs.crossover_methods.binomial_crossover import BinomialCrossover
from detpy.DETAlgs.data.alg_data import LShadeCnEpsinData
from detpy.DETAlgs.mutation_methods.current_to_pbest_1 import MutationCurrentToPBest1
from detpy.DETAlgs.random.index_generator import IndexGenerator
from detpy.DETAlgs.random.random_value_generator import RandomValueGenerator
from detpy.models.enums.boundary_constrain import fix_boundary_constraints_with_parent
from detpy.models.enums.optimization import OptimizationType
from detpy.models.population import Population


class LShadeCnEpsin(BaseAlg):
    """
    LSHADE-cnEpSin for continuous optimization.

    Extends LSHADE-EpSin with covariance matrix learning based crossover.
    """

    def __init__(self, params: LShadeCnEpsinData, db_conn=None, db_auto_write=False):
        super().__init__(LShadeCnEpsin.__name__, params, db_conn, db_auto_write)

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

        self._archive.extend(copy.deepcopy(self._pop.members))

        self._min_pop_size = params.minimum_population_size
        self._start_population_size = self.population_size

        self._f_sin_freg = params.f_sin_freq

        self.population_size_reduction_strategy = params.population_reduction_strategy
        self._max_generation = self._estimate_max_generation(
            self.nfe_max,
            self._start_population_size,
            self._min_pop_size
        )

        self._pc = params.covariance_probability
        self._ps = params.neighborhood_proportion

        self._cov_matrix = np.eye(self._pop.arg_num)

        self._TERMINAL = np.nan

        self._EPSILON = 0.00001

        self._k_index = 0
        self._index_gen = IndexGenerator()
        self._random_value_gen = RandomValueGenerator()
        self._binomial_crossing = BinomialCrossover()
        self._archive_reduction = ArchiveReduction()

        self._LP = params.learning_period
        self._generation = 0
        self._p_sinusoidal = [0.5, 0.5]
        self._ns_history: List[List[int]] = [[], []]
        self._nf_history: List[List[int]] = [[], []]
        self._ns_current = [0, 0]
        self._nf_current = [0, 0]
        self._EPSILON_PROB = 0.01

    def _estimate_max_generation(self, max_nfe: int, start_pop_size: int, min_pop_size: int):
        """
        Estimate Gmax used by the EpSin equations from the NFE budget and LPSR schedule.

        The paper defines EpSin using generation number g, while the stopping condition is FE-based.
        This mirrors the generation/NFE progression used in next_epoch.
        """
        generation = 0
        current_nfe = start_pop_size
        current_pop_size = start_pop_size

        while current_nfe < max_nfe:
            generation += 1
            current_nfe += current_pop_size
            current_pop_size = self.population_size_reduction_strategy.get_new_population_size(
                current_nfe,
                max_nfe,
                start_pop_size,
                min_pop_size
            )
            current_pop_size = max(min_pop_size, min(current_pop_size, start_pop_size))

        return max(1, generation)

    def _current_generation(self):
        return self._generation + 1

    def _is_first_half_generation(self):
        return self._current_generation() <= (self._max_generation / 2)

    def _get_member_vector(self, member):
        """Return decision vector of a member as numpy array."""
        return np.array([chromosome.real_value for chromosome in member.chromosomes], dtype=float)

    def _compute_neighborhood_indices(self):
        """
        Compute indices of individuals that belong to the Euclidean neighborhood of the best member.

        According to the paper, the neighborhood is formed around the current best individual based
        on Euclidean distance and contains NP * ps individuals.
        """
        num_neighbors = max(2, int(self._pop.size * self._ps))

        if self._pop.optimization == OptimizationType.MINIMIZATION:
            best_index = min(range(self._pop.size), key=lambda i: self._pop.members[i].fitness_value)
        else:
            best_index = max(range(self._pop.size), key=lambda i: self._pop.members[i].fitness_value)

        best_vec = self._get_member_vector(self._pop.members[best_index])

        distances = []
        for i, member in enumerate(self._pop.members):
            vec = self._get_member_vector(member)
            dist = np.linalg.norm(vec - best_vec)
            distances.append((i, dist))

        distances.sort(key=lambda t: t[1])
        neighborhood_indices = [idx for idx, _ in distances[:num_neighbors]]
        return neighborhood_indices

    def _update_covariance_matrix(self, neighborhood_indices):
        """
        Update covariance matrix C = B D B^T using individuals from the Euclidean neighborhood.
        Eq. (16).
        """
        if not neighborhood_indices:
            self._cov_matrix = np.eye(self._pop.arg_num)
            return

        population_matrix = np.array(
            [self._get_member_vector(self._pop.members[i]) for i in neighborhood_indices],
            dtype=float
        )

        cov = np.cov(population_matrix, rowvar=False)
        if np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
            self._cov_matrix = np.eye(self._pop.arg_num)
        else:
            self._cov_matrix = cov

    def _eig_crossover_internal(self, x_member, v_member, cr):
        """
        Perform covariance matrix learning based crossover using eigen decomposition.
        Eqs. (17)-(20).
        """
        x_values = self._get_member_vector(x_member)
        v_values = self._get_member_vector(v_member)

        try:
            _, B = np.linalg.eigh(self._cov_matrix)
        except np.linalg.LinAlgError:
            B = np.eye(len(x_values))

        xt = B.T @ x_values
        vt = B.T @ v_values

        randmask = (np.random.rand(len(x_values)) < cr)
        ut = np.where(randmask, vt, xt)

        jrand = np.random.randint(0, len(ut))
        ut[jrand] = vt[jrand]

        u_values = B @ ut

        new_member = copy.deepcopy(x_member)
        for i, value in enumerate(u_values):
            new_member.chromosomes[i].real_value = float(value)
        return new_member

    def _crossover_with_covariance(self, origin_population: Population, mutated_population: Population,
                                   cr_table: List[float]) -> Population:
        """
        Apply crossover: with probability pc use covariance matrix learning based crossover (Section B),
        otherwise use standard binomial crossover. The neighborhood is used only to compute the
        covariance matrix, not to restrict which individuals receive the eigenvector crossover.
        """
        neighborhood_indices = self._compute_neighborhood_indices()
        self._update_covariance_matrix(neighborhood_indices)

        new_members = []
        for i in range(origin_population.size):
            if np.random.rand() < self._pc:
                member = self._eig_crossover_internal(
                    origin_population.members[i],
                    mutated_population.members[i],
                    cr_table[i]
                )
            else:
                member = self._binomial_crossing.crossover_members(
                    origin_population.members[i],
                    mutated_population.members[i],
                    cr_table[i]
                )
            new_members.append(member)

        return Population.with_new_members(origin_population, new_members)

    def _update_population_size(self, nfe: int, max_nfe: int, start_pop_size: int, min_pop_size: int):
        """
        Calculate new population size using Linear Population Size Reduction (LPSR).
        Eq. (3).
        """
        new_size = self.population_size_reduction_strategy.get_new_population_size(
            nfe, max_nfe, start_pop_size, min_pop_size
        )

        self._pop.resize(new_size)

    def _mutate(self, population: Population, the_best_to_select_table: List[int], f_table: List[float]) -> Population:
        """
        Perform current-to-pbest/1 mutation. Eq. (2).
        """
        new_members = []

        pa = np.concatenate((population.members, self._archive))

        for i in range(population.size):
            r1 = self._index_gen.generate_unique(len(population.members), [i])
            r2 = self._index_gen.generate_unique(len(pa), [i, r1])

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
        return Population.with_new_members(population, new_members)

    def _selection(self, origin, modified, ftable, cr_table,
                   freg_values: Dict[int, float], config_used: Dict[int, int]):
        """
        Greedy selection. Track successful F, Cr, freq values and per-config success/failure counts.

        Parameters:
        - freg_values: maps individual index -> freq value (only for config 1 / adaptive sinusoidal).
        - config_used: maps individual index -> config id (0 or 1), only for first-half individuals.
        """
        new_members = []

        for i in range(origin.size):
            is_original_better = (
                (origin.optimization == OptimizationType.MINIMIZATION and origin.members[i] <= modified.members[i]) or
                (origin.optimization == OptimizationType.MAXIMIZATION and origin.members[i] >= modified.members[i])
            )

            if is_original_better:
                new_members.append(copy.deepcopy(origin.members[i]))
                if i in config_used:
                    self._nf_current[config_used[i]] += 1
            else:
                self._archive.append(copy.deepcopy(origin.members[i]))
                self._success_f.append(ftable[i])
                self._success_cr.append(cr_table[i])
                df = abs(modified.members[i].fitness_value - origin.members[i].fitness_value)
                self._difference_fitness_success.append(df)

                if i in freg_values:
                    self._difference_fitness_success_freg.append(df)
                    self._success_freg.append(freg_values[i])

                if i in config_used:
                    self._ns_current[config_used[i]] += 1

                new_members.append(copy.deepcopy(modified.members[i]))
        return Population.with_new_members(origin, new_members)

    def _reset_success_metrics(self):
        """Reset success metrics after memory update."""
        self._success_f = []
        self._success_cr = []
        self._success_freg = []
        self._difference_fitness_success = []
        self._difference_fitness_success_freg = []

    def _update_memory(self, success_f, success_cr, success_freg, df, df_freg):
        """
        Update memory for Cr, F, and freq based on successful trial vectors.
        Eqs. (11)-(15) -- weighted Lehmer mean.
        """

        is_first_half = self._is_first_half_generation()
        memory_updated = False

        if is_first_half:
            if len(success_freg) > 0:
                total = np.sum(df_freg)
                if not np.isclose(total, 0.0, atol=self._EPSILON):
                    weights = np.array(df_freg) / total
                    s = np.array(success_freg)
                    freg_new = self._weighted_lehmer_mean(s, weights)
                    freg_new = np.clip(freg_new, 0, 1)
                    random_index = int(np.random.randint(0, self._H))
                    self._freq_memory[random_index] = freg_new
                    memory_updated = True
            if len(success_f) > 0 and len(success_cr) > 0:
                total = np.sum(df)
                if np.isclose(total, 0.0, atol=self._EPSILON):
                    self._memory_Cr[self._k_index] = self._TERMINAL
                    memory_updated = True
                else:
                    weights = np.array(df) / total
                    s_cr = np.array(success_cr)
                    cr_new = self._weighted_lehmer_mean(s_cr, weights)
                    cr_new = np.clip(cr_new, 0, 1)
                    self._memory_Cr[self._k_index] = cr_new
                    memory_updated = True
        else:
            if len(success_f) > 0 and len(success_cr) > 0:
                total = np.sum(df)
                if np.isclose(total, 0.0, atol=self._EPSILON):
                    self._memory_Cr[self._k_index] = self._TERMINAL
                    memory_updated = True
                else:
                    weights = np.array(df) / total
                    s_cr = np.array(success_cr)
                    cr_new = self._weighted_lehmer_mean(s_cr, weights)
                    cr_new = np.clip(cr_new, 0, 1)
                    self._memory_Cr[self._k_index] = cr_new

                    s_f = np.array(success_f)
                    f_new = self._weighted_lehmer_mean(s_f, weights)
                    f_new = np.clip(f_new, 0, 1)
                    self._memory_F[self._k_index] = f_new
                    memory_updated = True

        if memory_updated:
            self._k_index = (self._k_index + 1) % self._H
        self._reset_success_metrics()

    @staticmethod
    def _weighted_lehmer_mean(values, weights):
        numerator = np.sum(weights * values * values)
        denominator = np.sum(weights * values)
        return numerator / denominator if denominator > 0 else 0.0

    def _update_sinusoidal_probabilities(self):
        """
        Update probabilities for the two sinusoidal configurations (Eqs. 7-8).
        Called once per generation. In the first LP generations p1 = p2 = 0.5.
        """
        self._generation += 1

        for k in range(2):
            self._ns_history[k].append(self._ns_current[k])
            self._nf_history[k].append(self._nf_current[k])

        self._ns_current = [0, 0]
        self._nf_current = [0, 0]

        for k in range(2):
            if len(self._ns_history[k]) > self._LP:
                self._ns_history[k].pop(0)
                self._nf_history[k].pop(0)

        if self._generation < self._LP:
            self._p_sinusoidal = [0.5, 0.5]
            return

        s = [0.0, 0.0]
        for k in range(2):
            sum_ns = sum(self._ns_history[k])
            sum_nf = sum(self._nf_history[k])
            total = sum_ns + sum_nf
            if total > 0:
                s[k] = sum_ns / total + self._EPSILON_PROB
            else:
                s[k] = self._EPSILON_PROB

        total_s = s[0] + s[1]
        if total_s > 0:
            self._p_sinusoidal = [s[0] / total_s, s[1] / total_s]
        else:
            self._p_sinusoidal = [0.5, 0.5]

    def _initialize_parameters_for_epoch(self):
        """
        Generate F, Cr, pbest count, freq values and config assignment for each individual.

        Returns:
        - f_table: scaling factors.
        - cr_table: crossover rates.
        - bests_to_select: number of p-best members per individual.
        - freg_values: dict mapping index -> freq value (only for adaptive sinusoidal config).
        - config_used: dict mapping index -> config id (0 = non-adaptive, 1 = adaptive).
        """
        f_table = []
        cr_table = []
        bests_to_select = []
        freg_values = {}
        config_used = {}

        for idx in range(self._pop.size):
            ri = self._index_gen.generate(0, self._H)

            if np.isnan(self._memory_Cr[ri]):
                cr = 0.0
            else:
                cr = self._random_value_gen.generate_normal(self._memory_Cr[ri], 0.1, 0.0, 1.0)

            is_first_half = self._is_first_half_generation()

            if is_first_half:
                generation = self._current_generation()
                if generation <= self._LP:
                    use_non_adaptive = np.random.rand() < 0.5
                else:
                    use_non_adaptive = self._p_sinusoidal[0] >= self._p_sinusoidal[1]
                if use_non_adaptive:
                    f = 0.5 * (math.sin(2 * math.pi * self._f_sin_freg * generation + math.pi) * (
                            self._max_generation - generation) / self._max_generation + 1.0)
                    config_used[idx] = 0
                else:
                    ri = np.random.randint(0, self._H)
                    elem_freg_external_memory = self._freq_memory[ri]

                    elem_cauchy = np.random.standard_cauchy() * 0.1 + elem_freg_external_memory
                    elem_cauchy = np.clip(elem_cauchy, 0.0, 1.0)

                    f = 0.5 * (math.sin(2 * math.pi * elem_cauchy * generation) * (
                            generation / self._max_generation) + 1.0)

                    freg_values[idx] = elem_cauchy
                    config_used[idx] = 1
            else:
                f = self._random_value_gen.generate_cauchy_greater_than_zero(self._memory_F[ri], 0.1, 0.0, 1.0)

            f_table.append(f)
            cr_table.append(cr)
            best_member_count = max(2, int(self._p * self._pop.size))
            best_member_count = min(best_member_count, self._pop.size)
            bests_to_select.append(best_member_count)

        return f_table, cr_table, bests_to_select, freg_values, config_used

    def next_epoch(self):
        """
        Perform one generation of the LSHADE-cnEpSin algorithm (Fig. 1 pseudocode).
        """
        f_table, cr_table, bests_to_select, freg_values, config_used = self._initialize_parameters_for_epoch()
        mutant = self._mutate(self._pop, bests_to_select, f_table)

        trial = self._crossover_with_covariance(self._pop, mutant, cr_table)
        fix_boundary_constraints_with_parent(self._pop, trial, self.boundary_constraints_fun)
        trial.update_fitness_values(self._function.eval, self.parallel_processing)

        self._pop = self._selection(self._pop, trial, f_table, cr_table, freg_values, config_used)
        self._archive = self._archive_reduction.reduce_archive(self._archive, self._pop.size, self._pop.size)
        self._update_memory(self._success_f, self._success_cr, self._success_freg, self._difference_fitness_success,
                            self._difference_fitness_success_freg)

        if self._is_first_half_generation():
            self._update_sinusoidal_probabilities()
        else:
            self._generation += 1

        self._update_population_size(self.nfe, self.nfe_max, self._start_population_size,
                                     self._min_pop_size)
