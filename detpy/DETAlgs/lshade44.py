import copy
from typing import Tuple, Dict

import numpy as np

from detpy.DETAlgs.archive_reduction.archive_reduction import ArchiveReduction
from detpy.DETAlgs.base import BaseAlg
from detpy.DETAlgs.crossover_methods.binomial_crossover import BinomialCrossover
from detpy.DETAlgs.crossover_methods.exponential_crossover import ExponentialCrossover
from detpy.DETAlgs.data.alg_data import LSHADE44Data
from detpy.DETAlgs.mutation_methods.current_to_pbest_1 import MutationCurrentToPBest1
from detpy.DETAlgs.mutation_methods.mutation_randrl_1 import MutationRandrl1
from detpy.DETAlgs.random.index_generator import IndexGenerator
from detpy.DETAlgs.random.random_value_generator import RandomValueGenerator
from detpy.DETAlgs.validator.params_validator import ParameterValidator
from detpy.models.enums.boundary_constrain import fix_boundary_constraints_with_parent
from detpy.models.enums.optimization import OptimizationType
from detpy.models.population import Population


class LSHADE44(BaseAlg):
    """
    LSHADE44: L-SHADE with Competing Strategies Applied to CEC2015
    Learning-based Test Suite.

    References:
    Radka Polakova, Josef Tvrdik, Petr Bujok
    University of Ostrava
    """

    STRATEGY_CB = 0
    STRATEGY_CE = 1
    STRATEGY_RB = 2
    STRATEGY_RE = 3

    STRATEGY_NAMES = {
        0: "current-to-pbest/bin (cb)",
        1: "current-to-pbest/exp (ce)",
        2: "randrl/1/bin (rb)",
        3: "randrl/1/exp (re)",
    }

    STRATEGY_SHORT_NAMES = {
        0: "cb",
        1: "ce",
        2: "rb",
        3: "re",
    }

    CROSSOVER_BINOMIAL = 0
    CROSSOVER_EXPONENTIAL = 1

    def __init__(
        self,
        params: LSHADE44Data,
        db_conn=None,
        db_auto_write=False,
        db_writing_interval=5000,
        verbose=False,
        monitor=None,
    ):
        super().__init__(
            LSHADE44.__name__,
            params,
            db_conn,
            db_auto_write,
            db_writing_interval,
            verbose,
            monitor,
        )

        ParameterValidator.int_min(
            params.population_size,
            10,
            "population_size",
        )

        ParameterValidator.positive_int(
            params.memory_size,
            "memory_size",
        )

        ParameterValidator.float_between(
            params.best_member_percentage,
            0.0,
            1.0,
            "best_member_percentage",
        )

        ParameterValidator.float_times_geq(
            params.best_member_percentage,
            params.population_size,
            1,
            "best_member_percentage",
        )

        ParameterValidator.int_min(
            params.smoothing_constant,
            2,
            "smoothing_constant",
        )

        ParameterValidator.positive_float(
            params.reset_threshold,
            "reset_threshold",
        )

        ParameterValidator.positive_int(
            params.minimum_population_size,
            "minimum_population_size",
        )

        ParameterValidator.int_times_leq(
            params.minimum_population_size,
            1,
            params.population_size,
            "minimum_population_size",
        )

        ParameterValidator.positive_int(
            params.archive_size,
            "archive_size",
        )

        ParameterValidator.positive_int(
            params.pm_to_cr_table_size,
            "pm_to_cr_table_size",
        )

        # ---------------------------------------------------------
        # Core parameters
        # ---------------------------------------------------------

        self._H = params.memory_size
        self._K = 4

        # Historical memories.
        self._memory_F = np.full(
            (self._K, self._H),
            0.5,
        )

        self._memory_Cr = np.full(
            (self._K, self._H),
            0.5,
        )

        # Memory pointers.
        self._k_indices = np.zeros(
            self._K,
            dtype=int,
        )

        # ---------------------------------------------------------
        # Success information
        # ---------------------------------------------------------

        self._successF = [[] for _ in range(self._K)]
        self._successCr = [[] for _ in range(self._K)]
        self._difference_fitness_success = [[] for _ in range(self._K)]

        # ---------------------------------------------------------
        # Strategy competition
        # ---------------------------------------------------------

        self._n0 = params.smoothing_constant
        self._delta = params.reset_threshold

        self._strategy_success_counts = np.zeros(
            self._K,
            dtype=float,
        )

        self._strategy_probabilities = np.full(
            self._K,
            1.0 / self._K,
        )

        # ---------------------------------------------------------
        # Archive
        # ---------------------------------------------------------

        self._archive_size = params.archive_size
        self._archive = []

        # ---------------------------------------------------------
        # Population size reduction
        # ---------------------------------------------------------

        self._min_pop_size = params.minimum_population_size
        self._start_population_size = self.population_size
        self._population_size_reduction_strategy = (
            params.population_reduction_strategy
        )

        # ---------------------------------------------------------
        # Algorithm parameters
        # ---------------------------------------------------------

        self._p = params.best_member_percentage

        # ---------------------------------------------------------
        # PM -> CR table
        # ---------------------------------------------------------

        self._d_p = params.pm_to_cr_table_size
        self._pm_to_cr_table = self._precompute_pm_to_cr_table()

        # ---------------------------------------------------------
        # Components
        # ---------------------------------------------------------

        self._index_gen = IndexGenerator()

        self._binomial_crossing = BinomialCrossover()
        self._exponential_crossing = ExponentialCrossover()

        self._archive_reduction = ArchiveReduction()
        self._random_value_gen = RandomValueGenerator()

        # ---------------------------------------------------------
        # Historical statistics
        # ---------------------------------------------------------

        self._strategy_usage_stats = []
        self._strategy_success_stats = []

        # ---------------------------------------------------------
        # Metrics for monitoring
        # ---------------------------------------------------------

        self._monitor_metrics = {}

    # ==================================================================
    # PM -> CR
    # ==================================================================

    def _precompute_pm_to_cr_table(self) -> Dict[float, float]:
        """
        Pre-compute table mapping pm -> CR for exponential crossover.

        Solves:

            CR^d - d*pm*CR + d*pm - 1 = 0

        If there are two solutions, the smaller one is stored.
        """

        table = {}

        d = self.nr_of_args

        pm_min = 1.0 / d
        pm_max = 1.0

        def f(cr, pm):
            return cr ** d - d * pm * cr + d * pm - 1

        for r in range(self._d_p + 1):
            pm = (
                pm_min
                + r * (pm_max - pm_min) / self._d_p
            )

            if abs(pm - pm_min) < 1e-15:
                table[pm] = 0.0
                continue

            if abs(pm - 1.0) < 1e-15:
                table[pm] = 1.0
                continue

            lo = 0.0
            hi = 1.0 - 1e-12

            flo = f(lo, pm)
            fhi = f(hi, pm)

            if flo * fhi > 0:
                cr = 1.0

            else:
                for _ in range(100):
                    mid = 0.5 * (lo + hi)
                    fm = f(mid, pm)

                    if abs(fm) < 1e-15:
                        lo = hi = mid
                        break

                    if flo * fm <= 0:
                        hi = mid
                    else:
                        lo = mid
                        flo = fm

                cr = 0.5 * (lo + hi)

            table[pm] = float(
                np.clip(cr, 0.0, 1.0)
            )

        return table

    def _pm_to_cr(self, pm: float) -> float:
        """
        Convert pm to CR using the pre-computed table.
        """

        pm = np.clip(
            pm,
            1.0 / self.nr_of_args,
            1.0,
        )

        closest_pm = min(
            self._pm_to_cr_table.keys(),
            key=lambda x: abs(x - pm),
        )

        return self._pm_to_cr_table[closest_pm]

    # ==================================================================
    # Strategy selection
    # ==================================================================

    def _select_strategy(self) -> int:
        """
        Select a DE strategy using roulette-wheel selection.
        """

        return np.random.choice(
            self._K,
            p=self._strategy_probabilities,
        )

    def _generate_parameters(
        self,
        strategy_idx: int,
    ) -> Tuple[float, float, int, float]:
        """
        Generate F, CR/pm and crossover type.
        """

        ri = self._index_gen.generate(
            0,
            self._H,
        )

        # ---------------------------------------------------------
        # Mutation factor F
        # ---------------------------------------------------------

        f = self._random_value_gen.generate_cauchy_greater_than_zero(
            mean=self._memory_F[strategy_idx, ri],
            scale=0.1,
            max_val=1.0,
        )

        # ---------------------------------------------------------
        # Crossover
        # ---------------------------------------------------------

        is_exponential = strategy_idx in [
            self.STRATEGY_CE,
            self.STRATEGY_RE,
        ]

        if is_exponential:

            pm = self._random_value_gen.generate_normal(
                mean=self._memory_Cr[strategy_idx, ri],
                std_dev=0.1,
                min_val=1.0 / self.nr_of_args,
                max_val=1.0,
            )

            cr = self._pm_to_cr(pm)

            crossover_type = self.CROSSOVER_EXPONENTIAL

        else:

            cr = self._random_value_gen.generate_normal(
                mean=self._memory_Cr[strategy_idx, ri],
                std_dev=0.1,
                min_val=0.0,
                max_val=1.0,
            )

            pm = None

            crossover_type = self.CROSSOVER_BINOMIAL

        return (
            f,
            cr,
            crossover_type,
            pm,
        )

    # ==================================================================
    # Memory update
    # ==================================================================

    def _update_memory_for_strategy(
        self,
        strategy_idx: int,
    ):
        """
        Update historical memory for one strategy.
        """

        if (
            len(self._successF[strategy_idx]) == 0
            or len(self._successCr[strategy_idx]) == 0
        ):
            return

        total_diff = np.sum(
            self._difference_fitness_success[strategy_idx]
        )

        if total_diff <= 0:
            return

        weights = (
            np.array(
                self._difference_fitness_success[strategy_idx]
            )
            / total_diff
        )

        # ---------------------------------------------------------
        # M_CR
        # ---------------------------------------------------------

        cr_new = np.sum(
            weights
            * np.array(self._successCr[strategy_idx])
        )

        cr_new = np.clip(
            cr_new,
            0,
            1,
        )

        self._memory_Cr[
            strategy_idx,
            self._k_indices[strategy_idx],
        ] = cr_new

        # ---------------------------------------------------------
        # M_F
        # ---------------------------------------------------------

        f_values = np.array(
            self._successF[strategy_idx]
        )

        f_num = np.sum(
            weights
            * f_values
            * f_values
        )

        f_den = np.sum(
            weights
            * f_values
        )

        if f_den > 0:

            f_new = f_num / f_den

            f_new = np.clip(
                f_new,
                0,
                1,
            )

            self._memory_F[
                strategy_idx,
                self._k_indices[strategy_idx],
            ] = f_new

        self._k_indices[strategy_idx] = (
            self._k_indices[strategy_idx] + 1
        ) % self._H

    def _update_memories(self):
        """
        Update memories for all strategies that had successes.
        """

        for k in range(self._K):

            if len(self._successF[k]) > 0:
                self._update_memory_for_strategy(k)

    # ==================================================================
    # Strategy probabilities
    # ==================================================================

    def _update_strategy_probabilities(self):
        """
        Update strategy selection probabilities.
        """

        total_successes = np.sum(
            self._strategy_success_counts
        )

        if total_successes == 0:
            return

        for k in range(self._K):

            self._strategy_probabilities[k] = (
                self._strategy_success_counts[k]
                + self._n0
            ) / (
                total_successes
                + self._K * self._n0
            )

        self._strategy_probabilities = (
            self._strategy_probabilities
            / np.sum(self._strategy_probabilities)
        )

        # Reset if any probability falls below delta.
        if np.any(
            self._strategy_probabilities < self._delta
        ):
            self._strategy_probabilities = np.full(
                self._K,
                1.0 / self._K,
            )

            return

        self._strategy_usage_stats.append(
            copy.deepcopy(
                self._strategy_probabilities
            )
        )

        self._strategy_success_stats.append(
            copy.deepcopy(
                self._strategy_success_counts
            )
        )

    # ==================================================================
    # Mutation
    # ==================================================================

    def _mutate(
        self,
        i: int,
        strategy_idx: int,
        f: float,
    ) -> np.ndarray:
        """
        Apply mutation according to selected strategy.
        """

        current_member = self._pop.members[i]

        if strategy_idx in [
            self.STRATEGY_CB,
            self.STRATEGY_CE,
        ]:

            r1 = self._index_gen.generate_unique(
                self._pop.size,
                [i],
            )

            archive_pop = (
                list(self._pop.members)
                + list(self._archive)
            )

            archive_len = len(archive_pop)

            r2 = self._index_gen.generate_unique(
                archive_len,
                [i, r1],
            )

            best_count = int(
                self._p * self._pop.size
            )

            best_members = self._pop.get_best_members(
                best_count
            )

            pbest_idx = self._index_gen.generate(
                0,
                len(best_members),
            )

            pbest = best_members[pbest_idx]

            mutant = MutationCurrentToPBest1.mutate(
                base_member=current_member,
                best_member=pbest,
                r1=self._pop.members[r1],
                r2=archive_pop[r2],
                f=f,
            )

        else:

            mutant = MutationRandrl1.mutate(
                population=self._pop,
                current_index=i,
                f=f,
                optimization=self._pop.optimization,
            )

        return mutant

    # ==================================================================
    # Monitoring metrics
    # ==================================================================

    def _build_monitor_metrics(
        self,
        strategy_indices,
        f_table,
        cr_table,
        pm_table,
        crossover_types,
        success_improvements,
    ) -> dict:
        """
        Build scalar metrics for TensorBoard / CSV monitoring.

        All returned values are scalars so they can be directly
        consumed by Monitor implementations.
        """

        metrics = {}

        # ---------------------------------------------------------
        # Strategy probabilities
        # ---------------------------------------------------------

        for k in range(self._K):

            name = self.STRATEGY_SHORT_NAMES[k]

            metrics[
                f"strategy_probability/{name}"
            ] = float(
                self._strategy_probabilities[k]
            )

        # ---------------------------------------------------------
        # Strategy usage / success
        # ---------------------------------------------------------

        usage_counts = np.zeros(
            self._K,
            dtype=int,
        )

        for k in range(self._K):
            usage_counts[k] = np.sum(
                strategy_indices == k
            )

        total_usage = np.sum(
            usage_counts
        )

        total_successes = np.sum(
            self._strategy_success_counts
        )

        metrics[
            "strategy/total_usage"
        ] = float(total_usage)

        metrics[
            "strategy/total_successes"
        ] = float(total_successes)

        for k in range(self._K):

            name = self.STRATEGY_SHORT_NAMES[k]

            usage = int(
                usage_counts[k]
            )

            successes = int(
                self._strategy_success_counts[k]
            )

            metrics[
                f"strategy_usage/{name}"
            ] = float(usage)

            metrics[
                f"strategy_success/{name}"
            ] = float(successes)

            # Success rate = successful trials / strategy uses.
            success_rate = (
                successes / usage
                if usage > 0
                else 0.0
            )

            metrics[
                f"strategy_success_rate/{name}"
            ] = float(success_rate)

            # Percentage of all strategy selections.
            usage_percentage = (
                usage / total_usage
                if total_usage > 0
                else 0.0
            )

            metrics[
                f"strategy_usage_percentage/{name}"
            ] = float(usage_percentage)

        # ---------------------------------------------------------
        # Fitness improvement per strategy
        # ---------------------------------------------------------

        for k in range(self._K):

            name = self.STRATEGY_SHORT_NAMES[k]

            improvements = success_improvements[k]

            if improvements:

                metrics[
                    f"strategy_improvement/{name}"
                ] = float(
                    np.mean(improvements)
                )

                metrics[
                    f"strategy_best_improvement/{name}"
                ] = float(
                    np.max(improvements)
                )

            else:

                metrics[
                    f"strategy_improvement/{name}"
                ] = 0.0

                metrics[
                    f"strategy_best_improvement/{name}"
                ] = 0.0

        # ---------------------------------------------------------
        # Parameters - all strategies
        # ---------------------------------------------------------

        if len(f_table) > 0:

            metrics[
                "parameters/mean_F"
            ] = float(np.mean(f_table))

            metrics[
                "parameters/std_F"
            ] = float(np.std(f_table))

            metrics[
                "parameters/min_F"
            ] = float(np.min(f_table))

            metrics[
                "parameters/max_F"
            ] = float(np.max(f_table))

        if len(cr_table) > 0:

            metrics[
                "parameters/mean_CR"
            ] = float(np.mean(cr_table))

            metrics[
                "parameters/std_CR"
            ] = float(np.std(cr_table))

            metrics[
                "parameters/min_CR"
            ] = float(np.min(cr_table))

            metrics[
                "parameters/max_CR"
            ] = float(np.max(cr_table))

        # ---------------------------------------------------------
        # Parameters per strategy
        # ---------------------------------------------------------

        for k in range(self._K):

            name = self.STRATEGY_SHORT_NAMES[k]

            mask = strategy_indices == k

            if not np.any(mask):
                continue

            strategy_f = f_table[mask]
            strategy_cr = cr_table[mask]

            metrics[
                f"parameters/{name}/mean_F"
            ] = float(
                np.mean(strategy_f)
            )

            metrics[
                f"parameters/{name}/std_F"
            ] = float(
                np.std(strategy_f)
            )

            metrics[
                f"parameters/{name}/mean_CR"
            ] = float(
                np.mean(strategy_cr)
            )

            metrics[
                f"parameters/{name}/std_CR"
            ] = float(
                np.std(strategy_cr)
            )

        # ---------------------------------------------------------
        # PM for exponential crossover
        # ---------------------------------------------------------

        valid_pm = pm_table[
            ~np.isnan(pm_table)
        ]

        if len(valid_pm) > 0:

            metrics[
                "parameters/mean_pm"
            ] = float(
                np.mean(valid_pm)
            )

            metrics[
                "parameters/std_pm"
            ] = float(
                np.std(valid_pm)
            )

            metrics[
                "parameters/min_pm"
            ] = float(
                np.min(valid_pm)
            )

            metrics[
                "parameters/max_pm"
            ] = float(
                np.max(valid_pm)
            )

        # ---------------------------------------------------------
        # Crossover usage
        # ---------------------------------------------------------

        binomial_count = np.sum(
            crossover_types
            == self.CROSSOVER_BINOMIAL
        )

        exponential_count = np.sum(
            crossover_types
            == self.CROSSOVER_EXPONENTIAL
        )

        total_crossovers = (
            binomial_count
            + exponential_count
        )

        metrics[
            "crossover/binomial_count"
        ] = float(binomial_count)

        metrics[
            "crossover/exponential_count"
        ] = float(exponential_count)

        if total_crossovers > 0:

            metrics[
                "crossover/binomial_percentage"
            ] = float(
                binomial_count
                / total_crossovers
            )

            metrics[
                "crossover/exponential_percentage"
            ] = float(
                exponential_count
                / total_crossovers
            )

        # ---------------------------------------------------------
        # Memory statistics
        # ---------------------------------------------------------

        metrics[
            "memory/mean_F"
        ] = float(
            np.mean(self._memory_F)
        )

        metrics[
            "memory/std_F"
        ] = float(
            np.std(self._memory_F)
        )

        metrics[
            "memory/mean_CR"
        ] = float(
            np.mean(self._memory_Cr)
        )

        metrics[
            "memory/std_CR"
        ] = float(
            np.std(self._memory_Cr)
        )

        # Per-strategy memory.
        for k in range(self._K):

            name = self.STRATEGY_SHORT_NAMES[k]

            metrics[
                f"memory/{name}/mean_F"
            ] = float(
                np.mean(self._memory_F[k])
            )

            metrics[
                f"memory/{name}/mean_CR"
            ] = float(
                np.mean(self._memory_Cr[k])
            )

        # ---------------------------------------------------------
        # Memory pointers
        # ---------------------------------------------------------

        for k in range(self._K):

            name = self.STRATEGY_SHORT_NAMES[k]

            metrics[
                f"memory/{name}/pointer"
            ] = float(
                self._k_indices[k]
            )

        # ---------------------------------------------------------
        # Population
        # ---------------------------------------------------------

        metrics[
            "population/size"
        ] = float(
            self._pop.size
        )

        metrics[
            "population/archive_size"
        ] = float(
            len(self._archive)
        )

        # ---------------------------------------------------------
        # Strategy state
        # ---------------------------------------------------------

        metrics[
            "strategy/active_count"
        ] = float(
            np.sum(
                usage_counts > 0
            )
        )

        metrics[
            "strategy/dominant"
        ] = float(
            np.argmax(
                self._strategy_probabilities
            )
        )

        return metrics

    def get_monitor_metrics(self) -> dict:
        """
        Return current algorithm-specific monitoring metrics.

        This method is intended to be called by BaseAlg after
        next_epoch() has completed.
        """

        return self._monitor_metrics.copy()

    # ==================================================================
    # Epoch
    # ==================================================================

    def next_epoch(self):
        """
        Perform the next epoch of LSHADE44.
        """

        # ---------------------------------------------------------
        # Reset success information
        # ---------------------------------------------------------

        for k in range(self._K):

            self._successF[k] = []
            self._successCr[k] = []
            self._difference_fitness_success[k] = []

        self._strategy_success_counts = np.zeros(
            self._K,
            dtype=float,
        )

        # ---------------------------------------------------------
        # Generate parameters
        # ---------------------------------------------------------

        population_size = self._pop.size

        f_table = np.zeros(
            population_size,
            dtype=float,
        )

        cr_table = np.zeros(
            population_size,
            dtype=float,
        )

        # IMPORTANT:
        # pm is None for binomial crossover.
        # Therefore NaN is used instead of None.
        pm_table = np.full(
            population_size,
            np.nan,
            dtype=float,
        )

        crossover_types = np.zeros(
            population_size,
            dtype=int,
        )

        strategy_indices = np.zeros(
            population_size,
            dtype=int,
        )

        # ---------------------------------------------------------
        # Select strategy and generate parameters
        # ---------------------------------------------------------

        for i in range(population_size):

            strategy = self._select_strategy()

            f, cr, crossover_type, pm = (
                self._generate_parameters(strategy)
            )

            strategy_indices[i] = strategy
            f_table[i] = f
            cr_table[i] = cr

            if pm is not None:
                pm_table[i] = pm

            crossover_types[i] = crossover_type

        # ---------------------------------------------------------
        # Generate trial vectors
        # ---------------------------------------------------------

        trial_members = [
            None
            for _ in range(population_size)
        ]

        for i in range(population_size):

            strategy_idx = strategy_indices[i]

            current_member = self._pop.members[i]

            # Mutation
            mutant = self._mutate(
                i,
                strategy_idx,
                f_table[i],
            )

            # Crossover
            if (
                crossover_types[i]
                == self.CROSSOVER_BINOMIAL
            ):

                trial = (
                    self._binomial_crossing.crossover_members(
                        current_member,
                        mutant,
                        cr_table[i],
                    )
                )

            else:

                trial = (
                    self._exponential_crossing.crossover_members(
                        current_member,
                        mutant,
                        cr_table[i],
                    )
                )

            trial_members[i] = trial

        # ---------------------------------------------------------
        # Create trial population
        # ---------------------------------------------------------

        trial_pop = Population.with_new_members(
            self._pop,
            trial_members,
        )

        # ---------------------------------------------------------
        # Boundary constraints
        # ---------------------------------------------------------

        fix_boundary_constraints_with_parent(
            self._pop,
            trial_pop,
            self.boundary_constraints_fun,
        )

        # ---------------------------------------------------------
        # Evaluate trial population
        # ---------------------------------------------------------

        trial_pop.update_fitness_values(
            self._function.eval,
            self.parallel_processing,
        )

        # ---------------------------------------------------------
        # Selection
        # ---------------------------------------------------------

        new_members = []

        # Temporary list for monitoring.
        success_improvements = [
            []
            for _ in range(self._K)
        ]

        for i in range(population_size):

            strategy = strategy_indices[i]

            origin_member = self._pop.members[i]
            trial_member = trial_pop.members[i]

            # -----------------------------------------------------
            # Compare fitness
            # -----------------------------------------------------

            if (
                self._pop.optimization
                == OptimizationType.MINIMIZATION
            ):

                is_better = (
                    trial_member.fitness_value
                    < origin_member.fitness_value
                )

            else:

                is_better = (
                    trial_member.fitness_value
                    > origin_member.fitness_value
                )

            # -----------------------------------------------------
            # Successful trial
            # -----------------------------------------------------

            if is_better:

                self._strategy_success_counts[
                    strategy
                ] += 1

                # Successful F
                self._successF[
                    strategy
                ].append(
                    f_table[i]
                )

                # Successful CR / pm
                if (
                    crossover_types[i]
                    == self.CROSSOVER_EXPONENTIAL
                ):

                    self._successCr[
                        strategy
                    ].append(
                        pm_table[i]
                    )

                else:

                    self._successCr[
                        strategy
                    ].append(
                        cr_table[i]
                    )

                # Fitness improvement
                improvement = abs(
                    origin_member.fitness_value
                    - trial_member.fitness_value
                )

                self._difference_fitness_success[
                    strategy
                ].append(
                    improvement
                )

                success_improvements[
                    strategy
                ].append(
                    improvement
                )

                # Archive
                self._archive.append(
                    copy.deepcopy(
                        origin_member
                    )
                )

                # Keep trial
                new_members.append(
                    copy.deepcopy(
                        trial_member
                    )
                )

            # -----------------------------------------------------
            # Failed trial
            # -----------------------------------------------------

            else:

                new_members.append(
                    copy.deepcopy(
                        origin_member
                    )
                )

        # ---------------------------------------------------------
        # Update population
        # ---------------------------------------------------------

        self._pop = Population.with_new_members(
            self._pop,
            new_members,
        )

        # ---------------------------------------------------------
        # Reduce archive
        # ---------------------------------------------------------

        self._archive = (
            self._archive_reduction.reduce_archive(
                self._archive,
                self._archive_size,
                self.population_size,
            )
        )

        # ---------------------------------------------------------
        # Update memories
        # ---------------------------------------------------------

        self._update_memories()

        # ---------------------------------------------------------
        # Update strategy probabilities
        # ---------------------------------------------------------

        self._update_strategy_probabilities()

        # ---------------------------------------------------------
        # Population size reduction
        # ---------------------------------------------------------

        self.update_population_size(
            self.nfe,
            self.nfe_max,
            self._start_population_size,
            self._min_pop_size,
        )

        # ---------------------------------------------------------
        # Build monitoring metrics LAST.
        #
        # This is important because:
        #
        # - memories are already updated
        # - strategy probabilities are already updated
        # - population size is already reduced
        # - archive size is already current
        # ---------------------------------------------------------

        self._monitor_metrics = (
            self._build_monitor_metrics(
                strategy_indices=strategy_indices,
                f_table=f_table,
                cr_table=cr_table,
                pm_table=pm_table,
                crossover_types=crossover_types,
                success_improvements=success_improvements,
            )
        )

    # ==================================================================
    # Population reduction
    # ==================================================================

    def update_population_size(
        self,
        nfe: int,
        total_nfe: int,
        start_pop_size: int,
        min_pop_size: int,
    ):
        """
        Calculate new population size using LPSR.
        """

        new_size = (
            self._population_size_reduction_strategy
            .get_new_population_size(
                nfe,
                total_nfe,
                start_pop_size,
                min_pop_size,
            )
        )

        self._pop.resize(
            new_size
        )

    # ==================================================================
    # Strategy statistics
    # ==================================================================

    def get_strategy_statistics(self) -> dict:
        """
        Compute statistics describing the usage and performance
        of all DE strategies.
        """

        if not self._strategy_usage_stats:

            return {
                "probabilities": [],
                "success_counts": [],
                "average_probabilities": np.zeros(
                    self._K
                ),
                "total_successes": np.zeros(
                    self._K
                ),
                "usage_percentage": np.zeros(
                    self._K
                ),
                "final_probabilities": (
                    self._strategy_probabilities.copy()
                ),
                "total_epochs": 0,
                "memory_F": self._memory_F.tolist(),
                "memory_Cr": self._memory_Cr.tolist(),
            }

        probs_array = np.array(
            self._strategy_usage_stats
        )

        success_array = np.array(
            self._strategy_success_stats
        )

        avg_probs = np.mean(
            probs_array,
            axis=0,
        )

        total_successes = np.sum(
            success_array,
            axis=0,
        )

        total_epochs = len(
            self._strategy_usage_stats
        )

        usage_percentage = (
            avg_probs * 100
        )

        return {
            "probabilities": self._strategy_usage_stats,
            "success_counts": self._strategy_success_stats,
            "average_probabilities": avg_probs,
            "total_successes": total_successes,
            "usage_percentage": usage_percentage,
            "final_probabilities": (
                self._strategy_probabilities.copy()
            ),
            "total_epochs": total_epochs,
            "memory_F": self._memory_F.tolist(),
            "memory_Cr": self._memory_Cr.tolist(),
        }

    def print_strategy_statistics(self):
        """
        Print detailed statistics about competing strategies.
        """

        stats = self.get_strategy_statistics()

        print("\n" + "=" * 80)
        print("STRATEGY STATISTICS FOR LSHADE44")
        print("=" * 80)

        print(
            "\n{:<25} {:>12} {:>15} {:>15} {:>12}".format(
                "Strategy",
                "Avg Prob %",
                "Total Success",
                "Usage %",
                "Final Prob %",
            )
        )

        print("-" * 80)

        for k in range(self._K):

            name = self.STRATEGY_NAMES[k]

            avg_prob = (
                stats["average_probabilities"][k]
                * 100
            )

            total_succ = (
                stats["total_successes"][k]
            )

            usage_pct = (
                stats["usage_percentage"][k]
            )

            final_prob = (
                stats["final_probabilities"][k]
                * 100
            )

            print(
                "{:<25} {:>12.2f} {:>15.0f} "
                "{:>15.2f} {:>12.2f}".format(
                    name,
                    avg_prob,
                    total_succ,
                    usage_pct,
                    final_prob,
                )
            )

        print("-" * 80)

        print(
            f"Total epochs: {stats['total_epochs']}"
        )

        print(
            f"Total successes: "
            f"{np.sum(stats['total_successes'])}"
        )

        # ---------------------------------------------------------
        # Memory
        # ---------------------------------------------------------

        print("\nMEMORY STATE PER STRATEGY:")
        print("-" * 80)

        for k in range(self._K):

            print(
                f"{self.STRATEGY_NAMES[k]}:"
            )

            print(
                f"  MF: "
                f"{stats['memory_F'][k][:5]}"
                f"... (first 5 values)"
            )

            print(
                f"  MC: "
                f"{stats['memory_Cr'][k][:5]}"
                f"... (first 5 values)"
            )

            print(
                f"  Pointer: "
                f"{self._k_indices[k]}"
            )

        self._print_probability_evolution(
            stats
        )

    def _print_probability_evolution(
        self,
        stats: dict,
        max_points: int = 20,
    ):
        """
        Display strategy probability evolution.
        """

        probs = stats["probabilities"]

        if not probs:
            return

        print(
            "\nPROBABILITY EVOLUTION "
            "(sampled every {} epochs):".format(
                max(
                    1,
                    len(probs) // max_points,
                )
            )
        )

        print("-" * 80)

        step = max(
            1,
            len(probs) // max_points,
        )

        sampled_indices = list(
            range(
                0,
                len(probs),
                step,
            )
        )

        header = "Epoch"

        for k in range(self._K):

            header += (
                f"  "
                f"{self.STRATEGY_SHORT_NAMES[k]:>6}"
            )

        print(header)
        print("-" * 80)

        for idx in sampled_indices[:max_points]:

            row = f"{idx:>5}"

            for k in range(self._K):

                row += (
                    f"  "
                    f"{probs[idx][k] * 100:>6.1f}"
                )

            print(row)

        if (
            sampled_indices[-1]
            != len(probs) - 1
        ):

            idx = len(probs) - 1

            row = f"{idx:>5}"

            for k in range(self._K):

                row += (
                    f"  "
                    f"{probs[idx][k] * 100:>6.1f}"
                )

            print(row)

        print("-" * 80)


    def get_detailed_strategy_report(self) -> dict:
        """
        Generate a detailed structured report.
        """

        stats = self.get_strategy_statistics()

        report = {}

        for k in range(self._K):

            report[
                self.STRATEGY_NAMES[k]
            ] = {
                "short_name":
                    self.STRATEGY_SHORT_NAMES[k],

                "average_probability":
                    stats[
                        "average_probabilities"
                    ][k],

                "total_successes":
                    int(
                        stats[
                            "total_successes"
                        ][k]
                    ),

                "usage_percentage":
                    stats[
                        "usage_percentage"
                    ][k],

                "final_probability":
                    stats[
                        "final_probabilities"
                    ][k],

                "memory_F":
                    stats[
                        "memory_F"
                    ][k],

                "memory_Cr":
                    stats[
                        "memory_Cr"
                    ][k],

                "memory_pointer":
                    int(
                        self._k_indices[k]
                    ),
            }

        report["total_epochs"] = (
            stats["total_epochs"]
        )

        report["total_successes"] = int(
            np.sum(
                stats["total_successes"]
            )
        )

        report["final_probabilities"] = (
            stats[
                "final_probabilities"
            ].tolist()
        )

        if (
            stats["total_epochs"] > 0
            and np.sum(
                stats["average_probabilities"]
            ) > 0
        ):

            report["dominant_strategy"] = (
                self.STRATEGY_NAMES[
                    np.argmax(
                        stats[
                            "average_probabilities"
                        ]
                    )
                ]
            )

        else:

            report["dominant_strategy"] = (
                "N/A (no data)"
            )

        return report