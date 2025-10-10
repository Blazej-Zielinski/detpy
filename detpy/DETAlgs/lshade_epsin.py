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

        self.H = params.memory_size
        self.memory_F = np.full(self.H, 0.5)
        self.memory_Cr = np.full(self.H, 0.5)

        self.success_cr = []
        self.success_f = []
        self.success_freg = []
        self.difference_fitness_success = []
        self.difference_fitness_success_freg = []

        self.P = params.init_probability_mutation_strategy

        self.freq_memory = np.full(self.H, 0.5)

        self.archive = []

        self.archive_size = self.population_size

        self.archive.extend(copy.deepcopy(self._pop.members))

        self.min_pop_size = params.minimum_population_size
        self.start_population_size = self.population_size

        self.freg = params.freg

        self.population_size_reduction_strategy = params.population_reduction_strategy

    def _update_population_size(self, nfe: int, max_nfe: int, start_pop_size: int, min_pop_size: int):

        new_size = self.population_size_reduction_strategy.get_new_population_size(
            nfe, max_nfe, start_pop_size, min_pop_size
        )

        self._pop.resize(new_size)

    def _mutate(self, population: Population, the_best_to_select_table: List[int], f_table: List[float]) -> Population:
        new_members = []

        pa = np.concatenate((population.members, self.archive))

        for i in range(population.size):
            r1 = np.random.choice(len(population.members))
            r2 = np.random.choice(len(pa))

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
        new_members = []

        for i in range(origin.size):
            if (origin.optimization == OptimizationType.MINIMIZATION and origin.members[i] <= modified.members[i]) or \
                    (origin.optimization == OptimizationType.MAXIMIZATION and origin.members[i] >= modified.members[i]):
                new_members.append(copy.deepcopy(origin.members[i]))
            else:
                self.archive.append(copy.deepcopy(origin.members[i]))
                self.success_f.append(ftable[i])
                self.success_cr.append(cr_table[i])
                df = abs(modified.members[i].fitness_value - origin.members[i].fitness_value)
                self.difference_fitness_success.append(df)

                if i in freg_values:
                    self.difference_fitness_success_freg.append(df)
                    self.success_freg.append(freg_values[i])

                new_members.append(copy.deepcopy(modified.members[i]))

        new_population = Population(
            lb=origin.lb, ub=origin.ub,
            arg_num=origin.arg_num, size=origin.size,
            optimization=origin.optimization
        )
        new_population.members = np.array(new_members)
        return new_population

    def _reset_success_metrics(self):
        self.success_f = []
        self.success_cr = []
        self.difference_fitness_success = []
        self.difference_fitness_success_freg = []

    def _update_memory(self, success_f, success_cr, df, df_freg):
        if not success_f:
            self._reset_success_metrics()
            return

        total = sum(df)
        if np.isclose(total, 0.0, atol=1e-12):
            self._reset_success_metrics()
            return

        weights = np.array(df, dtype=float) / float(total)

        denom_f = np.sum(weights * np.array(success_f, dtype=float))
        denom_cr = np.sum(weights * np.array(success_cr, dtype=float))

        if np.isclose(denom_f, 0.0, atol=1e-12) or np.isclose(denom_cr, 0.0, atol=1e-12):
            self._reset_success_metrics()
            return

        f_new = np.sum(weights * np.square(success_f)) / denom_f
        cr_new = np.sum(weights * np.square(success_cr)) / denom_cr

        f_new = float(np.clip(f_new, 0.0, 1.0))
        cr_new = float(np.clip(cr_new, 0.0, 1.0))

        self.memory_F = np.append(self.memory_F[1:], f_new)
        self.memory_Cr = np.append(self.memory_Cr[1:], cr_new)

        if len(self.success_freg) > 0:
            total_freg = sum(df_freg)
            if not np.isclose(total_freg, 0.0, atol=1e-12):
                weights_freg = np.array(df_freg, dtype=float) / float(total_freg)
                denom_freq = np.sum(weights_freg * np.array(self.success_freg, dtype=float))
                if not np.isclose(denom_freq, 0.0, atol=1e-12):
                    freq_new = np.sum(weights_freg * np.square(self.success_freg)) / denom_freq
                    freq_new = float(np.clip(freq_new, 0.01, 1.0))
                    self.freq_memory = np.append(self.freq_memory[1:], freq_new)
            self.success_freg = []

        self._reset_success_metrics()

    def _initialize_parameters_for_epoch(self):
        f_table = []
        cr_table = []
        bests_to_select = []
        freg_values = {}
        for idx in range(self._pop.size):
            ri = np.random.randint(0, self.H)
            cr = np.clip(np.random.normal(self.memory_Cr[ri], 0.1), 0.0, 1.0)

            is_lower_then_half_populations = self.nfe < (self.nfe_max / 2)

            if is_lower_then_half_populations:
                is_not_adaptive_sinusoidal_increasing = np.random.rand() < 0.5
                if is_not_adaptive_sinusoidal_increasing:
                    f = 0.5 * (math.sin(2 * math.pi * self.freg * self.nfe + math.pi) * (
                            self.nfe_max - self.nfe) / self.nfe_max + 1.0)
                else:

                    ri = np.random.randint(0, self.H)
                    elem_freg_external_memory = self.freq_memory[ri]

                    elem_cauchy = np.random.standard_cauchy() * 0.1 + elem_freg_external_memory
                    elem_cauchy = np.clip(elem_cauchy, 0.01, 1.0)  # nwm czy to potrzebne
                    freg_values[idx] = elem_cauchy
                    f = 0.5 * math.sin(2 * math.pi * elem_cauchy * self.nfe) * (
                            self.nfe / self.nfe_max) + 1
            else:
                while True:
                    f = np.random.standard_cauchy() * 0.1 + self.memory_F[ri]
                    if f > 0:
                        break

                if (f > 1):
                    f = 0.0

            f_table.append(f)
            cr_table.append(cr)
            bests_to_select.append(calculate_best_member_count(self.population_size))
        return f_table, cr_table, bests_to_select, freg_values

    def _perform_local_search(self):
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
        f_table, cr_table, bests_to_select, freg_values = self._initialize_parameters_for_epoch()
        mutant = self._mutate(self._pop, bests_to_select, f_table)

        trial = crossing(self._pop, mutant, cr_table)
        fix_boundary_constraints_with_parent(self._pop, trial, self.boundary_constraints_fun)
        trial.update_fitness_values(self._function.eval, self.parallel_processing)

        self._pop = self._selection(self._pop, trial, f_table, cr_table, freg_values)
        self.archive = archive_reduction(self.archive, self.archive_size, self.population_size)
        self._update_memory(self.success_f, self.success_cr, self.difference_fitness_success,
                            self.difference_fitness_success_freg)

        self._update_population_size(self.nfe, self.nfe_max, self.start_population_size,
                                     self.min_pop_size)

        if self._pop.size <= 20 and not hasattr(self, "_local_search_done"):
            self._local_search_done = True
            self._perform_local_search()

