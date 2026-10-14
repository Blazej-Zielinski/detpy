import copy
from collections import deque
from typing import List

import numpy as np

from detpy.DETAlgs.archive_reduction.archive_reduction import ArchiveReduction
from detpy.DETAlgs.base import BaseAlg
from detpy.DETAlgs.data.alg_data import LShadeCnEpsinData
from detpy.DETAlgs.methods.methods_lshade_cnepsin import (
    binomial_crossover_values,
    choose_larger_probability_config,
    configuration_probabilities,
    configuration_success_rate,
    covariance_eigenvectors,
    eigen_binomial_crossover,
    estimate_generation_max,
    euclidean_neighborhood_indices,
    is_better,
    member_values,
    set_member_values,
    sinusoidal_decreasing,
    sinusoidal_increasing,
    weighted_lehmer_mean,
)
from detpy.DETAlgs.mutation_methods.current_to_pbest_1 import MutationCurrentToPBest1
from detpy.DETAlgs.random.index_generator import IndexGenerator
from detpy.DETAlgs.random.random_value_generator import RandomValueGenerator
from detpy.DETAlgs.validator.params_validator import ParameterValidator
from detpy.models.enums.boundary_constrain import fix_boundary_constraints_with_parent
from detpy.models.enums.optimization import OptimizationType
from detpy.models.population import Population


class LSHADECnEpSin(BaseAlg):
    """
    LSHADE-cnEpSin

    Ensemble sinusoidal differential covariance matrix adaptation with Euclidean
    neighborhood, including linear population size reduction.

    Links:
    https://ieeexplore.ieee.org/document/7969336

    References:
    Awad, N. H., Ali, M. Z., & Suganthan, P. N. (2017). Ensemble sinusoidal differential
    covariance matrix adaptation with Euclidean neighborhood for solving CEC2017 benchmark
    problems. In 2017 IEEE Congress on Evolutionary Computation (CEC) (pp. 372–379). IEEE.
    """

    def __init__(self, params: LShadeCnEpsinData, db_conn=None, db_auto_write=False,
                 db_writing_interval=5000, verbose=False):
        super().__init__(LSHADECnEpSin.__name__, params, db_conn, db_auto_write, db_writing_interval, verbose)

        ParameterValidator.int_min(params.memory_size, 1, "memory_size")
        ParameterValidator.int_min(params.minimum_population_size, 4, "minimum_population_size")
        ParameterValidator.min_max(params.minimum_population_size, params.population_size, "Population size")
        ParameterValidator.float_between(params.best_member_percentage, 0.0, 1.0, "best_member_percentage")
        ParameterValidator.float_times_geq(params.best_member_percentage, params.population_size, 1, "best_member_percentage")
        ParameterValidator.float_between(params.f_init, 0.0, 1.0, "f_init")
        ParameterValidator.float_between(params.cr_init, 0.0, 1.0, "cr_init")
        ParameterValidator.float_between(params.freq_init, 0.0, 1.0, "freq_init")
        ParameterValidator.float_between(params.freq_decreasing, 0.0, 1.0, "freq_decreasing")
        ParameterValidator.float_between(params.covariance_probability, 0.0, 1.0, "covariance_probability")
        ParameterValidator.float_between(params.neighborhood_fraction, 0.0, 1.0, "neighborhood_fraction")
        ParameterValidator.int_min(params.learning_period, 1, "learning_period")

        self._H = params.memory_size
        self._memory_F = np.full(self._H, params.f_init, dtype=float)
        self._memory_Cr = np.full(self._H, params.cr_init, dtype=float)
        self._memory_freq = np.full(self._H, params.freq_init, dtype=float)

        self._p = params.best_member_percentage
        self._pc = params.covariance_probability
        self._ps = params.neighborhood_fraction
        self._freq_decreasing = params.freq_decreasing
        self._learning_period = params.learning_period

        self._prob_sin = [0.5, 0.5]
        self._ns_history = deque(maxlen=self._learning_period)
        self._nf_history = deque(maxlen=self._learning_period)

        self._k_index = 0
        self._success_f = []
        self._success_cr = []
        self._success_freq = []
        self._difference_fitness_success = []
        self._difference_fitness_success_freq = []

        self._archive = list(copy.deepcopy(self._pop.members))
        self._min_pop_size = params.minimum_population_size
        self._start_population_size = self.population_size
        self._population_size_reduction_strategy = params.population_reduction_strategy
        self._generation_max = estimate_generation_max(
            self.nfe_max, self._start_population_size, self._min_pop_size
        )

        self._SUCCESS_EPS = 0.01

        self._generation = 0
        self._last_cml_count = 0
        self._last_binomial_count = 0

        self._index_gen = IndexGenerator()
        self._random_value_gen = RandomValueGenerator()
        self._archive_reduction = ArchiveReduction()

    def _current_generation(self) -> int:
        return self._generation + 1

    def _in_first_half(self) -> bool:
        return self._current_generation() <= (self._generation_max / 2.0)

    def _pbest_count(self) -> int:
        return max(2, int(self._pop.size * self._p)) if self._pop.size >= 2 else 1

    def _reset_success_metrics(self):
        self._success_f = []
        self._success_cr = []
        self._success_freq = []
        self._difference_fitness_success = []
        self._difference_fitness_success_freq = []

    def _refresh_sin_probabilities(self, generation: int):
        if generation <= self._learning_period:
            self._prob_sin = [0.5, 0.5]
            return
        if not self._ns_history:
            self._prob_sin = [0.5, 0.5]
            return

        ns = list(zip(*self._ns_history))
        nf = list(zip(*self._nf_history))
        rates = [
            configuration_success_rate(ns[0], nf[0], self._SUCCESS_EPS),
            configuration_success_rate(ns[1], nf[1], self._SUCCESS_EPS),
        ]
        self._prob_sin = configuration_probabilities(rates)

    def _sample_cr(self, memory_cr: float) -> float:
        return self._random_value_gen.generate_normal(memory_cr, 0.1, 0.0, 1.0)

    def _sample_freq(self) -> float:
        mean_freq = self._memory_freq[self._index_gen.generate(0, self._H)]
        while True:
            freq = np.random.standard_cauchy() * 0.1 + mean_freq
            if freq > 0:
                break
        return float(np.clip(freq, 0.0, 1.0))

    def _initialize_parameters_for_epoch(self):
        f_table = []
        cr_table = []
        freq_table = []
        config_table = []
        pbest_counts = []

        generation = self._current_generation()
        in_first_half = self._in_first_half()
        config = None
        decreasing_f = None

        if in_first_half:
            self._refresh_sin_probabilities(generation)
            config = choose_larger_probability_config(self._prob_sin)
            if config == 0:
                decreasing_f = float(np.clip(
                    sinusoidal_decreasing(self._freq_decreasing, generation, self._generation_max),
                    0.0, 1.0
                ))

        for _ in range(self._pop.size):
            ri = self._index_gen.generate(0, self._H)
            cr = self._sample_cr(self._memory_Cr[ri])

            freq_i = None
            if in_first_half:
                if config == 0:
                    f = decreasing_f
                else:
                    freq_i = self._sample_freq()
                    f = float(np.clip(
                        sinusoidal_increasing(freq_i, generation, self._generation_max),
                        0.0, 1.0
                    ))
            else:
                f = self._random_value_gen.generate_cauchy_greater_than_zero(
                    self._memory_F[ri], 0.1, 0.0, 1.0
                )

            f_table.append(f)
            cr_table.append(cr)
            freq_table.append(freq_i)
            config_table.append(config)
            pbest_counts.append(self._pbest_count())

        return f_table, cr_table, freq_table, config_table, pbest_counts

    def _mutate(self, f_table: List[float], pbest_counts: List[int]) -> Population:
        new_members = []
        archive_union = np.concatenate((self._pop.members, self._archive)) if self._archive else self._pop.members

        for i in range(self._pop.size):
            r1 = self._index_gen.generate_unique(self._pop.size, [i])
            r2 = self._index_gen.generate_unique(len(archive_union), [i, r1])
            best_members = self._pop.get_best_members(min(pbest_counts[i], self._pop.size))
            pbest = best_members[self._index_gen.generate(0, len(best_members))]
            mutant = MutationCurrentToPBest1.mutate(
                base_member=self._pop.members[i],
                best_member=pbest,
                r1=self._pop.members[r1],
                r2=archive_union[r2],
                f=f_table[i]
            )
            new_members.append(mutant)

        return Population.with_new_members(self._pop, new_members)

    def _crossover(self, mutant: Population, cr_table: List[float]) -> Population:
        points = np.array([member_values(member) for member in self._pop.members])
        if self._pop.optimization == OptimizationType.MINIMIZATION:
            best_index = int(np.argmin([member.fitness_value for member in self._pop.members]))
        else:
            best_index = int(np.argmax([member.fitness_value for member in self._pop.members]))

        neighborhood = euclidean_neighborhood_indices(points, best_index, self._ps)
        eigenvectors = covariance_eigenvectors(points[neighborhood])

        new_members = []
        self._last_cml_count = 0
        self._last_binomial_count = 0

        for i in range(self._pop.size):
            target = member_values(self._pop.members[i])
            donor = member_values(mutant.members[i])
            if np.random.rand() < self._pc:
                trial_values = eigen_binomial_crossover(target, donor, cr_table[i], eigenvectors)
                self._last_cml_count += 1
            else:
                trial_values = binomial_crossover_values(target, donor, cr_table[i])
                self._last_binomial_count += 1

            trial_member = copy.deepcopy(self._pop.members[i])
            set_member_values(trial_member, trial_values)
            new_members.append(trial_member)

        return Population.with_new_members(self._pop, new_members)

    def _selection(self, origin: Population, modified: Population, f_table, cr_table,
                   freq_table, config_table) -> Population:
        ns = [0, 0]
        nf = [0, 0]
        new_members = []

        for i in range(origin.size):
            orig = origin.members[i]
            mod = modified.members[i]
            config = config_table[i]

            if not is_better(orig.fitness_value, mod.fitness_value, origin.optimization):
                new_members.append(copy.deepcopy(orig))
                if config is not None:
                    nf[config] += 1
                continue

            self._archive.append(copy.deepcopy(orig))
            delta = abs(mod.fitness_value - orig.fitness_value)
            self._success_f.append(f_table[i])
            self._success_cr.append(cr_table[i])
            self._difference_fitness_success.append(delta)
            if freq_table[i] is not None:
                self._success_freq.append(freq_table[i])
                self._difference_fitness_success_freq.append(delta)
            if config is not None:
                ns[config] += 1
            new_members.append(copy.deepcopy(mod))

        if self._in_first_half():
            self._ns_history.append(ns)
            self._nf_history.append(nf)

        return Population.with_new_members(origin, new_members)

    def _update_memory(self):
        if self._success_f and self._success_cr:
            cr_new = weighted_lehmer_mean(self._success_cr, self._difference_fitness_success)
            self._memory_Cr[self._k_index] = float(np.clip(cr_new, 0.0, 1.0))

            if not self._in_first_half():
                f_new = weighted_lehmer_mean(self._success_f, self._difference_fitness_success)
                self._memory_F[self._k_index] = float(np.clip(f_new, 0.0, 1.0))

            if self._success_freq:
                freq_new = weighted_lehmer_mean(self._success_freq, self._difference_fitness_success_freq)
                random_index = self._index_gen.generate(0, self._H)
                self._memory_freq[random_index] = float(np.clip(freq_new, 0.0, 1.0))

            self._k_index = (self._k_index + 1) % self._H

        self._reset_success_metrics()

    def _update_population_size(self):
        new_size = self._population_size_reduction_strategy.get_new_population_size(
            self.nfe, self.nfe_max, self._start_population_size, self._min_pop_size
        )
        if new_size < self._pop.size:
            self._pop.resize(new_size)
        self._archive = self._archive_reduction.reduce_archive(
            self._archive, self._pop.size, self._pop.size
        )

    def next_epoch(self):
        f_table, cr_table, freq_table, config_table, pbest_counts = self._initialize_parameters_for_epoch()
        mutant = self._mutate(f_table, pbest_counts)
        trial = self._crossover(mutant, cr_table)
        fix_boundary_constraints_with_parent(self._pop, trial, self.boundary_constraints_fun)
        trial.update_fitness_values(self._function.eval, self.parallel_processing)

        self._pop = self._selection(self._pop, trial, f_table, cr_table, freq_table, config_table)
        self._update_memory()
        self._update_population_size()
        self._generation += 1
