import copy
from random import randrange

import numpy as np

from detpy.DETAlgs.base import BaseAlg
from detpy.DETAlgs.crossover_methods.binomial_crossover import BinomialCrossover
from detpy.DETAlgs.data.alg_data import DShadeData
from detpy.DETAlgs.methods.methods_dshade import (
    archive_capacity,
    dpsr_stage,
    population_size_after_reductions,
    reduce_population_to,
    weighted_lehmer_mean,
)
from detpy.DETAlgs.mutation_methods.current_to_pbest_1 import MutationCurrentToPBest1
from detpy.DETAlgs.random.index_generator import IndexGenerator
from detpy.DETAlgs.random.random_value_generator import RandomValueGenerator
from detpy.DETAlgs.validator.params_validator import ParameterValidator
from detpy.models.enums.boundary_constrain import fix_boundary_constraints_with_parent
from detpy.models.enums.optimization import OptimizationType
from detpy.models.population import Population


class DSHADE(BaseAlg):
    """
    D-SHADE

    SHADE 1.1 with Dynamic Population Size Reduction (DPSR) instead of the
    linear population size reduction used by L-SHADE.

    Links:
    https://ieeexplore.ieee.org/document/6900380

    References:
    Tanabe, R., & Fukunaga, A. S. (2014). Improving the search performance of
    SHADE using linear population size reduction. In 2014 IEEE Congress on
    Evolutionary Computation (CEC) (pp. 1658–1665). IEEE.
    https://doi.org/10.1109/cec.2014.6900380

    Brest, J., & Maučec, M. S. (2008). Population size reduction for the
    differential evolution algorithm. Applied Intelligence, 29(3), 228–247.
    """

    def __init__(self, params: DShadeData, db_conn=None, db_auto_write=False,
                 db_writing_interval=5000, verbose=False):
        super().__init__(DSHADE.__name__, params, db_conn, db_auto_write, db_writing_interval, verbose)

        ParameterValidator.int_min(params.memory_size, 1, "memory_size")
        ParameterValidator.int_min(params.r_n_init, 1, "r_n_init")
        ParameterValidator.int_min(params.reduction_frequency, 1, "reduction_frequency")
        ParameterValidator.int_min(params.minimum_population_size, 4, "minimum_population_size")
        ParameterValidator.min_max(params.minimum_population_size, params.population_size, "Population size")
        ParameterValidator.non_negative(params.r_arc, "r_arc")
        ParameterValidator.float_between(params.best_member_percentage, 0.0, 1.0, "best_member_percentage")

        self._H = params.memory_size
        self._memory_F = np.full(self._H, 0.5, dtype=float)
        self._memory_Cr = np.full(self._H, 0.5, dtype=float)
        self._k_index = 0
        self._terminal = np.nan

        self._p = params.best_member_percentage
        self._r_arc = params.r_arc
        self._r_n_init = params.r_n_init
        self._rfreq = params.reduction_frequency
        self._min_pop_size = params.minimum_population_size
        self._start_population_size = self.population_size

        self._archive = []
        self._success_cr = []
        self._success_f = []
        self._difference_fitness_success = []

        self._index_gen = IndexGenerator()
        self._random_value_gen = RandomValueGenerator()
        self._binomial_crossing = BinomialCrossover()

    def _sample_cr(self, memory_index: int) -> float:
        entry = self._memory_Cr[memory_index]
        if np.isnan(entry):
            return 0.0
        return float(self._random_value_gen.generate_normal(entry, 0.1, 0.0, 1.0))

    def _sample_f(self, memory_index: int) -> float:
        return float(self._random_value_gen.generate_cauchy_greater_than_zero(
            self._memory_F[memory_index], 0.1, 0.0, 1.0
        ))

    def _pbest_count(self) -> int:
        return max(1, int(round(self._pop.size * self._p)))

    def _trim_archive(self):
        capacity = archive_capacity(self.population_size, self._r_arc)
        while len(self._archive) > capacity:
            self._archive.pop(randrange(len(self._archive)))

    def _mutate(self, f_table) -> Population:
        new_members = []
        pool = np.concatenate((self._pop.members, np.array(self._archive, dtype=object)))
        pbest_count = self._pbest_count()

        for i in range(self._pop.size):
            r1 = int(self._index_gen.generate_unique(self._pop.size, [i]))
            r2 = int(self._index_gen.generate_unique(len(pool), [i, r1]))
            best_members = self._pop.get_best_members(pbest_count)
            best_index = int(self._index_gen.generate(0, len(best_members)))

            mutant = MutationCurrentToPBest1.mutate(
                base_member=self._pop.members[i],
                best_member=best_members[best_index],
                r1=self._pop.members[r1],
                r2=pool[r2],
                f=f_table[i],
            )
            new_members.append(mutant)

        return Population.with_new_members(self._pop, new_members)

    def _selection(self, trial: Population, f_table, cr_table) -> Population:
        minimization = self._pop.optimization == OptimizationType.MINIMIZATION
        new_members = []

        for i in range(self._pop.size):
            parent = self._pop.members[i]
            child = trial.members[i]
            parent_fitness = parent.fitness_value
            child_fitness = child.fitness_value

            if minimization:
                survives = child_fitness <= parent_fitness
                improved = child_fitness < parent_fitness
            else:
                survives = child_fitness >= parent_fitness
                improved = child_fitness > parent_fitness

            if improved:
                self._archive.append(copy.deepcopy(parent))
                self._success_cr.append(cr_table[i])
                self._success_f.append(f_table[i])
                self._difference_fitness_success.append(abs(child_fitness - parent_fitness))

            new_members.append(copy.deepcopy(child if survives else parent))

        return Population.with_new_members(self._pop, new_members)

    def _update_memory(self):
        success_cr = self._success_cr
        success_f = self._success_f
        deltas = self._difference_fitness_success
        if len(success_cr) == 0 or len(success_f) == 0:
            return

        slot = self._k_index
        if np.isnan(self._memory_Cr[slot]) or max(success_cr) == 0:
            self._memory_Cr[slot] = self._terminal
        else:
            self._memory_Cr[slot] = float(np.clip(weighted_lehmer_mean(success_cr, deltas), 0.0, 1.0))

        self._memory_F[slot] = float(np.clip(weighted_lehmer_mean(success_f, deltas), 0.0, 1.0))
        self._k_index = (slot + 1) % self._H

    def _apply_dpsr(self):
        stage = dpsr_stage(self.nfe, self.nfe_max, self._rfreq)
        target = population_size_after_reductions(self._start_population_size, stage, self._min_pop_size)
        if target < self._pop.size:
            reduced = reduce_population_to(
                self._pop.members, target, self._min_pop_size, self._pop.optimization
            )
            self._pop.members = np.array(reduced, dtype=object)
            self._pop.size = len(reduced)
            self.population_size = self._pop.size
        self._trim_archive()

    def next_epoch(self):
        self._success_cr = []
        self._success_f = []
        self._difference_fitness_success = []
        self.population_size = self._pop.size

        f_table = []
        cr_table = []
        for _ in range(self._pop.size):
            memory_index = int(self._index_gen.generate(0, self._H))
            cr_table.append(self._sample_cr(memory_index))
            f_table.append(self._sample_f(memory_index))

        mutant = self._mutate(f_table)
        fix_boundary_constraints_with_parent(self._pop, mutant, self.boundary_constraints_fun)
        trial = self._binomial_crossing.crossover_population(self._pop, mutant, cr_table)
        trial.update_fitness_values(self._function.eval, self.parallel_processing)

        self._pop = self._selection(trial, f_table, cr_table)
        self._trim_archive()

        self._update_memory()
        self._apply_dpsr()
