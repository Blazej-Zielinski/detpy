import copy

import numpy as np
from typing import List

from detpy.DETAlgs.archive_reduction.archive_reduction import ArchiveReduction
from detpy.DETAlgs.base import BaseAlg
from detpy.DETAlgs.crossover_methods.binomial_crossover import BinomialCrossover
from detpy.DETAlgs.data.alg_data import LShadeCnEpsinData
from detpy.DETAlgs.methods.methods_lshade_cnepsin import (
    configuration_probabilities,
    configuration_success_rate,
    covariance_eigenvectors,
    eigen_binomial_crossover,
    estimate_generation_max,
    euclidean_neighborhood_indices,
    linear_population_size,
    member_values,
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

    LSHADE-EpSin with two changes from Awad, Ali and Suganthan (2017):
    the sinusoidal scaling factor is chosen from its recent success rates,
    and crossover is done in the covariance eigenbasis with probability pc.

    Mutation current-to-pbest, the external archive and the CR memory follow
    LSHADE-EpSin. The population size still falls linearly, and Eq. 3 rounds
    a half up to the next integer. The Gaussian-walk local search is not part
    of the algorithm given in the article.

    References:
    Awad, N. H., Ali, M. Z., & Suganthan, P. N. (2017). Ensemble sinusoidal
    differential covariance matrix adaptation with Euclidean neighborhood for
    solving CEC2017 benchmark problems. In 2017 IEEE Congress on Evolutionary
    Computation (CEC). IEEE.
    """

    def __init__(self, params: LShadeCnEpsinData, db_conn=None, db_auto_write=False, db_writing_interval=5000,
                 verbose=False, monitor=None):
        super().__init__(LSHADECnEpSin.__name__, params, db_conn, db_auto_write, db_writing_interval, verbose, monitor)

        ParameterValidator.int_min(params.memory_size, 1, "memory_size")
        ParameterValidator.int_min(params.minimum_population_size, 4, "minimum_population_size")
        ParameterValidator.int_times_leq(
            params.minimum_population_size, 1, params.population_size, "minimum_population_size"
        )
        ParameterValidator.positive_int(params.learning_period, "learning_period")
        ParameterValidator.float_between(params.best_member_percentage, 0.0, 1.0, "best_member_percentage")
        ParameterValidator.float_times_geq(
            params.best_member_percentage, params.population_size, 1, "best_member_percentage"
        )
        ParameterValidator.float_between(params.f_init, 0.0, 1.0, "f_init")
        ParameterValidator.float_between(params.cr_init, 0.0, 1.0, "cr_init")
        ParameterValidator.float_between(params.freq_init, 0.0, 1.0, "freq_init")
        ParameterValidator.non_negative(params.freq_decreasing, "freq_decreasing")
        ParameterValidator.float_between(params.covariance_probability, 0.0, 1.0, "covariance_probability")
        ParameterValidator.float_between(params.neighborhood_fraction, 0.0, 1.0, "neighborhood_fraction")

        self._H = params.memory_size
        self._memory_F = np.full(self._H, params.f_init)
        self._memory_Cr = np.full(self._H, params.cr_init)
        self._memory_freq = np.full(self._H, params.freq_init)

        self._success_f = []
        self._success_cr = []
        self._success_freq = []
        self._difference_fitness_success = []
        self._difference_fitness_success_freq = []
        self._ns_history = []
        self._nf_history = []
        self._prob_sin = [0.5, 0.5]

        self._p = params.best_member_percentage
        self._pc = params.covariance_probability
        self._ps = params.neighborhood_fraction
        self._freq_decreasing = params.freq_decreasing
        self._learning_period = params.learning_period

        self._archive = [copy.deepcopy(member) for member in self._pop.members]
        self._min_pop_size = params.minimum_population_size
        self._start_population_size = self.population_size
        self._generation = 0
        self._generation_max = estimate_generation_max(
            self.nfe_max, self._start_population_size, self._min_pop_size
        )

        # Same tolerance as LSHADE-EpSin for a fitness difference that is too small to weight.
        self._EPSILON = 0.00001
        self._k_index = 0
        self._index_gen = IndexGenerator()
        self._random_value_gen = RandomValueGenerator()
        self._binomial_crossing = BinomialCrossover()
        self._archive_reduction = ArchiveReduction()

    def _current_generation(self):
        return self._generation

    def _in_first_half(self):
        """First half of the generations uses the sinusoidal ensemble, g <= Gmax / 2."""
        return self._generation <= (self._generation_max / 2.0)

    def _pbest_count(self):
        return max(1, int(self._p * self._pop.size))

    def _update_population_size(self):
        """Reduce the population with Eq. 3 and drop the worst individuals."""
        new_size = linear_population_size(
            self.nfe, self.nfe_max, self._start_population_size, self._min_pop_size
        )
        self._pop.resize(new_size)
        self.population_size = self._pop.size

    def _sinusoidal_configuration(self):
        """
        Choose one sinusoidal configuration for this generation.

        For the first LP generations both configurations have probability 0.5
        and one is drawn at random. Later the configuration with the larger
        success probability, Eqs. 7-8, is used.
        """
        if self._current_generation() <= self._learning_period or not self._ns_history:
            self._prob_sin = [0.5, 0.5]
        else:
            rates = [
                configuration_success_rate(
                    [record[config] for record in self._ns_history],
                    [record[config] for record in self._nf_history],
                )
                for config in (0, 1)
            ]
            self._prob_sin = [float(probability) for probability in configuration_probabilities(rates)]

        if self._prob_sin[0] > self._prob_sin[1]:
            return 0
        if self._prob_sin[1] > self._prob_sin[0]:
            return 1
        return int(self._index_gen.generate(0, 2))

    def _initialize_parameters_for_epoch(self):
        """
        Build F, CR, frequency, configuration and p-best count for every individual.

        In the first half of the generations one sinusoidal configuration is
        used for the whole generation. In the second half F comes from the
        Cauchy memory and no frequency is adapted.
        """
        generation = self._current_generation()
        population_size = self._pop.size
        best_count = self._pbest_count()
        f_table = []
        cr_table = []
        freq_table = []
        use_sinusoid = self._in_first_half()
        config = self._sinusoidal_configuration() if use_sinusoid else None

        for _ in range(population_size):
            memory_index = self._index_gen.generate(0, self._H)
            cr = self._random_value_gen.generate_normal(self._memory_Cr[memory_index], 0.1, 0.0, 1.0)

            if not use_sinusoid:
                scaling_factor = self._random_value_gen.generate_cauchy_greater_than_zero(
                    self._memory_F[memory_index], 0.1, 0.0, 1.0
                )
                frequency = None
            elif config == 0:
                scaling_factor = sinusoidal_decreasing(
                    self._freq_decreasing, generation, self._generation_max
                )
                frequency = None
            else:
                frequency_index = self._index_gen.generate(0, self._H)
                frequency = float(np.clip(
                    np.random.standard_cauchy() * 0.1 + self._memory_freq[frequency_index],
                    0.0,
                    1.0,
                ))
                scaling_factor = sinusoidal_increasing(frequency, generation, self._generation_max)

            f_table.append(float(scaling_factor))
            cr_table.append(float(cr))
            freq_table.append(frequency)

        config_table = [config] * population_size
        pbest_counts = [best_count] * population_size
        return f_table, cr_table, freq_table, config_table, pbest_counts

    def _mutate(self, population: Population, the_best_to_select_table: List[int], f_table: List[float]) -> Population:
        """Current-to-pbest/1 mutation with the external archive, Eq. 2."""
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

    def _crossover(self, population: Population, mutant: Population, cr_table: List[float]) -> Population:
        """
        Crossover in the original coordinates, or in the eigen coordinates with probability pc.

        The eigenbasis is built once from the Euclidean neighborhood of the best individual.
        """
        if self._pc <= 0.0:
            new_members = [
                self._binomial_crossing.crossover_members(
                    population.members[i], mutant.members[i], cr_table[i]
                )
                for i in range(population.size)
            ]
            return Population.with_new_members(population, new_members)

        points = None
        basis = None
        new_members = []
        for i in range(population.size):
            if np.random.rand() < self._pc:
                if basis is None:
                    points = np.array([member_values(member) for member in population.members], dtype=float)
                    best_index = self._best_index(population)
                    neighborhood = euclidean_neighborhood_indices(points, best_index, self._ps)
                    basis = covariance_eigenvectors(points[neighborhood])
                trial_values = eigen_binomial_crossover(
                    points[i], member_values(mutant.members[i]), cr_table[i], basis
                )
                new_members.append(self._member_from_values(population.members[i], trial_values))
            else:
                new_members.append(self._binomial_crossing.crossover_members(
                    population.members[i], mutant.members[i], cr_table[i]
                ))
        return Population.with_new_members(population, new_members)

    def _best_index(self, population: Population) -> int:
        fitness = [member.fitness_value for member in population.members]
        if population.optimization == OptimizationType.MINIMIZATION:
            return int(np.argmin(fitness))
        return int(np.argmax(fitness))

    @staticmethod
    def _member_from_values(template, values):
        member = copy.deepcopy(template)
        for index, value in enumerate(values):
            member.chromosomes[index].real_value = float(value)
        return member

    def _selection(self, origin, modified, f_table, cr_table, freq_table, config_table):
        """
        Keep the parent when the trial is not strictly better.

        A replaced parent enters the archive. Successful F, CR and frequencies
        are stored for the memory update. Trial wins and losses of each
        sinusoidal configuration are counted for Eqs. 7-8.
        """
        new_members = []
        successes = [0, 0]
        failures = [0, 0]
        record_configurations = any(config is not None for config in config_table)

        for i in range(origin.size):
            if origin.optimization == OptimizationType.MINIMIZATION:
                improved = modified.members[i].fitness_value < origin.members[i].fitness_value
            else:
                improved = modified.members[i].fitness_value > origin.members[i].fitness_value

            config = config_table[i]
            if improved:
                self._archive.append(copy.deepcopy(origin.members[i]))
                self._success_f.append(f_table[i])
                self._success_cr.append(cr_table[i])
                difference = abs(modified.members[i].fitness_value - origin.members[i].fitness_value)
                self._difference_fitness_success.append(difference)
                if freq_table[i] is not None:
                    self._success_freq.append(freq_table[i])
                    self._difference_fitness_success_freq.append(difference)
                if record_configurations and config is not None:
                    successes[config] += 1
                new_members.append(copy.deepcopy(modified.members[i]))
            else:
                if record_configurations and config is not None:
                    failures[config] += 1
                new_members.append(copy.deepcopy(origin.members[i]))

        if record_configurations:
            self._ns_history.append(successes)
            self._nf_history.append(failures)
            extra = len(self._ns_history) - self._learning_period
            if extra > 0:
                del self._ns_history[:extra]
                del self._nf_history[:extra]

        return Population.with_new_members(origin, new_members)

    def _reset_success_metrics(self):
        self._success_f = []
        self._success_cr = []
        self._success_freq = []
        self._difference_fitness_success = []
        self._difference_fitness_success_freq = []

    def _lehmer_update(self, values, deltas):
        if len(values) == 0:
            return None
        values = np.asarray(values, dtype=float)
        deltas = np.asarray(deltas, dtype=float)
        total = float(np.sum(deltas))
        if total <= self._EPSILON:
            return None
        # Eqs. 13-14 are undefined when every successful value has zero weight.
        if float(np.sum((deltas / total) * values)) <= self._EPSILON:
            return None
        updated = float(weighted_lehmer_mean(values, deltas))
        if not np.isfinite(updated):
            return None
        return float(np.clip(updated, 0.0, 1.0))

    def _update_memory(self):
        """
        Write successful means back into the memories, Eqs. 11-15.

        F and CR move forward through memory index k. The frequency mean,
        which is only collected for the adaptive sinusoid, is written at a
        random memory position.
        """
        scaling_factor = self._lehmer_update(self._success_f, self._difference_fitness_success)
        crossover_rate = self._lehmer_update(self._success_cr, self._difference_fitness_success)
        if scaling_factor is not None and crossover_rate is not None:
            self._memory_F[self._k_index] = scaling_factor
            self._memory_Cr[self._k_index] = crossover_rate
            self._k_index = (self._k_index + 1) % self._H

        frequency = self._lehmer_update(self._success_freq, self._difference_fitness_success_freq)
        if frequency is not None:
            random_index = self._index_gen.generate(0, self._H)
            self._memory_freq[random_index] = frequency

        self._reset_success_metrics()

    def next_epoch(self):
        """Run one generation: adapt parameters, mutate, cross, select, then reduce NP."""
        self._generation += 1
        f_table, cr_table, freq_table, config_table, pbest_counts = self._initialize_parameters_for_epoch()
        mutant = self._mutate(self._pop, pbest_counts, f_table)
        trial = self._crossover(self._pop, mutant, cr_table)
        fix_boundary_constraints_with_parent(self._pop, trial, self.boundary_constraints_fun)
        trial.update_fitness_values(self._function.eval, self.parallel_processing)

        self._pop = self._selection(self._pop, trial, f_table, cr_table, freq_table, config_table)
        self._update_memory()
        self._update_population_size()
        self._archive = self._archive_reduction.reduce_archive(self._archive, self._pop.size)
