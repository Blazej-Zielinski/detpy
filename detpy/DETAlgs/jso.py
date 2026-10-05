import copy
from typing import List

import numpy as np

from detpy.DETAlgs.archive_reduction.archive_reduction import ArchiveReduction
from detpy.DETAlgs.base import BaseAlg
from detpy.DETAlgs.crossover_methods.binomial_crossover import BinomialCrossover
from detpy.DETAlgs.data.alg_data import JSOData
from detpy.DETAlgs.math.math_functions import MathFunctions
from detpy.DETAlgs.mutation_methods.current_to_pbest_r import MutationCurrentToPBestR
from detpy.DETAlgs.random.index_generator import IndexGenerator
from detpy.DETAlgs.random.random_value_generator import RandomValueGenerator
from detpy.DETAlgs.validator.params_validator import ParameterValidator
from detpy.models.enums.boundary_constrain import fix_boundary_constraints_with_parent
from detpy.models.enums.ilshade_p_update import ILShadePUpdateStrategy
from detpy.models.enums.optimization import OptimizationType

from detpy.models.population import Population


class JSO(BaseAlg):
    """
        jSO: Single Objective Real-Parameter Optimization

        An improved variant of iL-SHADE with a weighted current-to-pBest-w/1 mutation strategy.

        References:
        Brest, J., Sepesy Maučec, M., & Boškovič, B. (2017). Single objective real-parameter optimization: Algorithm jSO.
        In 2017 IEEE Congress on Evolutionary Computation (CEC) (pp. 1311–1318). IEEE.
    """

    def __init__(self, params: JSOData, db_conn=None, db_auto_write=False, db_writing_interval=5000, verbose=False, monitor=None):
        super().__init__(JSO.__name__, params, db_conn, db_auto_write, db_writing_interval, verbose, monitor)

        ParameterValidator.positive_int(
            params.memory_size,
            "memory_size"
        )

        ParameterValidator.float_between(
            params.p_max,
            0.0,
            1.0,
            "p_max"
        )

        ParameterValidator.float_between(
            params.p_min,
            0.0,
            1.0,
            "p_min"
        )

        ParameterValidator.min_max(
            params.p_min,
            params.p_max,
            "p"
        )

        self._H = params.memory_size
        self._memory_F = np.full(self._H, 0.5)
        self._memory_Cr = np.full(self._H, 0.8)
        self._memory_F[-1] = 0.9
        self._memory_Cr[-1] = 0.9

        self._p_max = params.p_max
        self._p_min = params.p_min
        self._p_update_strategy = params.p_update_strategy
        self._p = self._p_min
        self._k_index = 0

        self._successCr = []
        self._successF = []
        self._difference_fitness_success = []

        self._archive_size = self.population_size
        self._archive = []

        self._min_pop_size = params.minimum_population_size
        self._start_population_size = self.population_size
        self._population_size_reduction_strategy = params.population_reduction_strategy

        self._EPSILON = 0.00001

        self._index_gen = IndexGenerator()
        self._random_value_gen = RandomValueGenerator()
        self._binomial_crossing = BinomialCrossover()
        self._archive_reduction = ArchiveReduction()

    def _compute_fw(self, f: float) -> float:
        """
        Compute the weighted scaling factor Fw for current-to-pBest-w/1.

        Fw = 0.7 * F  if nfes < 0.2 * max_nfes
        Fw = 0.8 * F  if nfes < 0.4 * max_nfes
        Fw = 1.2 * F  otherwise
        """
        progress = self.nfe / self.nfe_max
        if progress < 0.2:
            return 0.7 * f
        if progress < 0.4:
            return 0.8 * f
        return 1.2 * f

    def _update_p(self):
        """
        Update p for current-to-pBest-w/1.

        Default (increasing):
            p = ((p_max - p_min) / max_nfes) * nfes + p_min
        """
        if self._p_update_strategy == ILShadePUpdateStrategy.DECREASING:
            self._p = self._p_max - ((self._p_max - self._p_min) / self.nfe_max) * self.nfe
        else:
            self._p = ((self._p_max - self._p_min) / self.nfe_max) * self.nfe + self._p_min

    def update_population_size(self, nfe: int, total_nfe: int, start_pop_size: int, min_pop_size: int):
        """
        Calculate new population size using Linear Population Size Reduction (LPSR).
        """
        new_size = self._population_size_reduction_strategy.get_new_population_size(
            nfe, total_nfe, start_pop_size, min_pop_size
        )
        self._pop.resize(new_size)

    def mutate(self,
               population: Population,
               the_best_to_select_table: List[int],
               f_table: List[float],
               fw_table: List[float]
               ) -> Population:
        """
        Perform mutation using current-to-pBest-w/1.

        v = x + Fw * (x_pBest - x) + F * (x_r1 - x_r2)
        """
        new_members = []
        sum_archive_and_population = np.concatenate((population.members, self._archive))

        for i in range(population.size):
            r1 = self._index_gen.generate_unique(len(population.members), [i])
            r2 = self._index_gen.generate_unique(len(sum_archive_and_population), [i, r1])

            best_members = population.get_best_members(the_best_to_select_table[i])
            random_index = self._index_gen.generate(0, len(best_members))
            selected_best_member = best_members[random_index]

            mutated_member = MutationCurrentToPBestR.mutate(
                base_member=population.members[i],
                best_member=selected_best_member,
                r1=population.members[r1],
                r2=sum_archive_and_population[r2],
                f=f_table[i],
                fw=fw_table[i],
            )

            new_members.append(mutated_member)

        return Population.with_new_members(population, new_members)

    def _selection(self, origin_population, modified_population, ftable, cr_table):
        """
        Selection:

        - replace parent when trial is not worse (f(u) <= f(x) for minimization)
        - store in archive / SF / SCR only on strict improvement (f(u) < f(x))
        """
        optimization = origin_population.optimization
        new_members = []

        if optimization == OptimizationType.MINIMIZATION:
            accepts = lambda orig, mod: mod.fitness_value <= orig.fitness_value
            is_strict_improvement = lambda orig, mod: mod.fitness_value < orig.fitness_value
            diff = lambda orig, mod: orig.fitness_value - mod.fitness_value
        else:
            accepts = lambda orig, mod: mod.fitness_value >= orig.fitness_value
            is_strict_improvement = lambda orig, mod: mod.fitness_value > orig.fitness_value
            diff = lambda orig, mod: mod.fitness_value - orig.fitness_value

        for i in range(origin_population.size):
            orig = origin_population.members[i]
            mod = modified_population.members[i]

            if not accepts(orig, mod):
                new_members.append(copy.deepcopy(orig))
                continue

            if is_strict_improvement(orig, mod):
                self._archive.append(copy.deepcopy(orig))
                self._successF.append(ftable[i])
                self._successCr.append(cr_table[i])
                self._difference_fitness_success.append(diff(orig, mod))

            new_members.append(copy.deepcopy(mod))

        return Population.with_new_members(origin_population, new_members)

    def update_memory(self, success_f: List[float], success_cr: List[float], difference_fitness_success: List[float]):
        """
        Update historical memory for F and Cr using the average of weighted Lehmer mean and previous value.
        """
        if len(success_f) == 0 or len(success_cr) == 0:
            return

        total = np.sum(difference_fitness_success)
        weights = difference_fitness_success / total
        old_cr = self._memory_Cr[self._k_index]
        old_f = self._memory_F[self._k_index]

        if old_cr < 0 or np.max(success_cr) == 0 or np.isclose(total, 0.0, atol=self._EPSILON):
            self._memory_Cr[self._k_index] = 0.0
        else:
            cr_new = MathFunctions.calculate_lehmer_mean(np.array(success_cr), weights, p=2)
            cr_new = np.clip((cr_new + old_cr) / 2, 0, 1)
            self._memory_Cr[self._k_index] = cr_new

        f_new = MathFunctions.calculate_lehmer_mean(np.array(success_f), weights, p=2)
        f_new = np.clip((f_new + old_f) / 2, 0, 1)
        self._memory_F[self._k_index] = f_new

        self._successF = []
        self._successCr = []
        self._difference_fitness_success = []
        self._k_index = (self._k_index + 1) % (self._H - 1)

    def _apply_early_stage_constraints(self, f: float, cr: float) -> tuple[float, float]:
        """
        Apply jSO early-stage constraints based on nfe progress.
        """
        progress = self.nfe / self.nfe_max

        if progress < 0.25:
            cr = max(cr, 0.7)
        elif progress < 0.5:
            cr = max(cr, 0.6)

        if progress < 0.6 and f > 0.7:
            f = 0.7

        return f, cr

    def initialize_parameters_for_epoch(self):
        """
        Initialize F, Cr, Fw, and p-best parameters for the next epoch.

        Returns:
        - f_table: Scaling factors for mutation.
        - cr_table: Crossover rates.
        - fw_table: Weighted scaling factors for the pBest term.
        - the_bests_to_select: Number of p-best members to select for each individual.
        """
        f_table = []
        cr_table = []
        fw_table = []
        the_bests_to_select = []

        for _ in range(self._pop.size):
            ri = np.random.randint(0, self._H)
            mean_f, mean_cr = self._memory_F[ri], self._memory_Cr[ri]

            if mean_cr < 0:
                cr = 0.0
            else:
                cr = self._random_value_gen.generate_normal(mean_cr, 0.1, 0.0, 1.0)

            f = self._random_value_gen.generate_cauchy_greater_than_zero(mean_f, 0.1, 0.0, 1.0)
            f, cr = self._apply_early_stage_constraints(f, cr)

            f_table.append(f)
            cr_table.append(cr)
            fw_table.append(self._compute_fw(f))
            the_bests_to_select.append(max(2, int(self.population_size * self._p)))

        return f_table, cr_table, fw_table, the_bests_to_select

    def next_epoch(self):
        """
        Perform the next epoch of the jSO algorithm.
        """
        self._successF = []
        self._successCr = []
        self._difference_fitness_success = []

        f_table, cr_table, fw_table, the_bests_to_select = self.initialize_parameters_for_epoch()

        mutant = self.mutate(self._pop, the_bests_to_select, f_table, fw_table)
        trial = self._binomial_crossing.crossover_population(self._pop, mutant, cr_table)

        fix_boundary_constraints_with_parent(self._pop, trial, self.boundary_constraints_fun)
        trial.update_fitness_values(self._function.eval, self.parallel_processing)

        new_pop = self._selection(self._pop, trial, f_table, cr_table)

        self._archive_size = self.population_size
        self._archive = self._archive_reduction.reduce_archive(self._archive, self._archive_size, self.population_size)

        self._pop = new_pop
        self.update_memory(self._successF, self._successCr, self._difference_fitness_success)

        self.update_population_size(self.nfe, self.nfe_max, self._start_population_size, self._min_pop_size)
        self._update_p()
