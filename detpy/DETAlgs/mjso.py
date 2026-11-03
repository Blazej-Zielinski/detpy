import copy
from typing import List

import numpy as np

from detpy.DETAlgs.base import BaseAlg
from detpy.DETAlgs.crossover_methods.binomial_crossover import BinomialCrossover
from detpy.DETAlgs.data.alg_data import MJSOData
from detpy.DETAlgs.math.math_functions import MathFunctions
from detpy.DETAlgs.mutation_methods.current_to_pbest_r import MutationCurrentToPBestR
from detpy.DETAlgs.random.index_generator import IndexGenerator
from detpy.DETAlgs.random.random_value_generator import RandomValueGenerator
from detpy.DETAlgs.validator.params_validator import ParameterValidator
from detpy.models.enums.boundary_constrain import fix_boundary_constraints_with_parent
from detpy.models.enums.ilshade_p_update import ILShadePUpdateStrategy
from detpy.models.enums.optimization import OptimizationType

from detpy.models.population import Population


class MJSO(BaseAlg):
    """
        MjSO

        jSO with three changes from Li et al. (2023):
        - xpbest is drawn from the top Npbest individuals with rank-based
          probabilities, so a better individual is more likely to be chosen
        - the difference vector points from the worse random individual toward
          the better one
        - the external archive is not used, and xr2 comes from the current
          population

        Parameter control, crossover, selection, historical memory and linear
        population-size reduction stay as in jSO.

        References:
        Li, Y., Han, T., Wang, X., Zhou, H., Tang, S., Huang, C., & Han, B. (2023).
        MjSO: A modified differential evolution with a probability selection
        mechanism and a directed mutation strategy.
        Swarm and Evolutionary Computation, 78, 101294.

        Brest, J., Sepesy Maučec, M., & Boškovič, B. (2017). Single objective
        real-parameter optimization: Algorithm jSO.
        In 2017 IEEE Congress on Evolutionary Computation (CEC) (pp. 1311–1318). IEEE.
    """

    def __init__(self, params: MJSOData, db_conn=None, db_auto_write=False, db_writing_interval=5000, verbose=False):
        super().__init__(MJSO.__name__, params, db_conn, db_auto_write, db_writing_interval, verbose)

        # The last memory slot stays at 0.9, so at least one other slot must exist.
        ParameterValidator.int_min(
            params.memory_size,
            2,
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

        if isinstance(params.m, bool) or not isinstance(params.m, (int, float)) or params.m < 1:
            raise ValueError(
                f"m must be a number >= 1, got {params.m!r}."
            )

        # xr1 and xr2 are both taken from the current population and must differ
        # from each other and from the target, so the population cannot shrink below 3.
        ParameterValidator.int_min(
            params.minimum_population_size,
            3,
            "minimum_population_size"
        )

        if params.minimum_population_size > params.population_size:
            raise ValueError(
                "minimum_population_size must be <= population_size, "
                f"got {params.minimum_population_size} > {params.population_size}."
            )

        self._H = params.memory_size
        self._memory_F = np.full(self._H, 0.5)
        self._memory_Cr = np.full(self._H, 0.8)
        self._memory_F[-1] = 0.9
        self._memory_Cr[-1] = 0.9

        self._m = params.m
        self._p_max = params.p_max
        self._p_min = params.p_min
        self._p_update_strategy = params.p_update_strategy
        self._p = self._p_min
        self._k_index = 0

        self._successCr = []
        self._successF = []
        self._difference_fitness_success = []

        self._min_pop_size = params.minimum_population_size
        self._start_population_size = self.population_size
        self._population_size_reduction_strategy = params.population_reduction_strategy

        self._EPSILON = 0.00001

        self._index_gen = IndexGenerator()
        self._random_value_gen = RandomValueGenerator()
        self._binomial_crossing = BinomialCrossover()

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

    def _selection_probabilities(self, count: int) -> np.ndarray:
        """
        Rank-based probabilities for the Npbest individuals, best to worst.

        Ri = m - (m - 1) * (i - 1) / (Npbest - 1)
        Pri = Ri / sum(Ri)
        """
        if count < 2:
            return np.ones(count)

        rank_offset = np.arange(count)
        weights = self._m - (self._m - 1.0) * rank_offset / (count - 1)
        return weights / np.sum(weights)

    def _select_pbest(self, best_members):
        probabilities = self._selection_probabilities(len(best_members))
        index = int(np.random.choice(len(best_members), p=probabilities))
        return best_members[index]

    def _directed_pair(self, first, second, optimization):
        """
        Order the pair so the mutation difference points toward the better member.

        Minimization uses first - second when f(first) <= f(second).
        Maximization uses the opposite comparison.
        """
        if optimization == OptimizationType.MINIMIZATION:
            first_is_better_or_equal = first.fitness_value <= second.fitness_value
        else:
            first_is_better_or_equal = first.fitness_value >= second.fitness_value

        if first_is_better_or_equal:
            return first, second
        return second, first

    def mutate(self,
               population: Population,
               the_best_to_select_table: List[int],
               f_table: List[float],
               fw_table: List[float]
               ) -> Population:
        """
        Perform directed current-to-pBest-w/1 mutation.

        xpbest is sampled from the top individuals with rank-based probabilities.
        xr1 and xr2 are distinct members of the current population, different from xi.
        The difference uses the better member minus the worse one.
        """
        new_members = []

        for i in range(population.size):
            r1 = self._index_gen.generate_unique(len(population.members), [i])
            r2 = self._index_gen.generate_unique(len(population.members), [i, r1])

            best_members = population.get_best_members(the_best_to_select_table[i])
            selected_best_member = self._select_pbest(best_members)
            leading, trailing = self._directed_pair(
                population.members[r1],
                population.members[r2],
                population.optimization,
            )

            mutated_member = MutationCurrentToPBestR.mutate(
                base_member=population.members[i],
                best_member=selected_best_member,
                r1=leading,
                r2=trailing,
                f=f_table[i],
                fw=fw_table[i],
            )

            new_members.append(mutated_member)

        return Population.with_new_members(population, new_members)

    def _selection(self, origin_population, modified_population, ftable, cr_table):
        """
        Selection:

        - replace parent when trial is not worse (f(u) <= f(x) for minimization)
        - store SF / SCR only on strict improvement (f(u) < f(x))
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
        Apply early-stage constraints based on nfe progress.
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
            the_bests_to_select.append(max(2, int(self._pop.size * self._p)))

        return f_table, cr_table, fw_table, the_bests_to_select

    def next_epoch(self):
        """
        Perform the next epoch of MjSO.
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

        self._pop = new_pop
        self.update_memory(self._successF, self._successCr, self._difference_fitness_success)

        self.update_population_size(self.nfe, self.nfe_max, self._start_population_size, self._min_pop_size)
        self._update_p()
