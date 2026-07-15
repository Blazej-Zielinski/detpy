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
from detpy.models.enums.boundary_constrain import fix_boundary_constraints_with_parent
from detpy.models.enums.optimization import OptimizationType
from detpy.models.population import Population


class LSHADE44(BaseAlg):
    """
    LSHADE44: L-SHADE with Competing Strategies Applied to CEC2015 Learning-based Test Suite

    References:
    Radka Polakova, Josef Tvrdik, Petr Bujok
    University of Ostrava
    """

    STRATEGY_CB = 0  # current-to-pbest/bin
    STRATEGY_CE = 1  # current-to-pbest/exp
    STRATEGY_RB = 2  # randrl/1/bin
    STRATEGY_RE = 3  # randrl/1/exp

    STRATEGY_NAMES = {
        0: 'current-to-pbest/bin (cb)',
        1: 'current-to-pbest/exp (ce)',
        2: 'randrl/1/bin (rb)',
        3: 'randrl/1/exp (re)'
    }

    STRATEGY_SHORT_NAMES = {
        0: 'cb',
        1: 'ce',
        2: 'rb',
        3: 're'
    }

    # Crossover types
    CROSSOVER_BINOMIAL = 0
    CROSSOVER_EXPONENTIAL = 1

    def __init__(self, params: LSHADE44Data, db_conn=None, db_auto_write=False):
        super().__init__(LSHADE44.__name__, params, db_conn, db_auto_write)

        MIN_POP_SIZE_FOR_PBEST = 10
        if self.population_size < MIN_POP_SIZE_FOR_PBEST:
            raise ValueError(
                "LSHADE44 requires population_size >= 10 "
                "because p is sampled from [2/NP, 0.2]."
            )

        self._H = params.memory_size  # Memory size for F and CR adaptation (per strategy)
        self._K = 4  # Number of strategies

        # Four pairs of historical circle memories - one pair for each strategy
        # Each pair consists of MF (for F adaptation) and MC (for CR/pm adaptation)
        self._memory_F = np.full((self._K, self._H), 0.5)  # shape: (4, H)
        self._memory_Cr = np.full((self._K, self._H), 0.5)  # shape: (4, H)

        # Pointers for each strategy's memory
        self._k_indices = np.zeros(self._K, dtype=int)  # shape: (4,)

        # Success sets for each strategy
        self._successF = [[] for _ in range(self._K)]
        self._successCr = [[] for _ in range(self._K)]
        self._difference_fitness_success = [[] for _ in range(self._K)]

        # Strategy competition mechanism
        self._n0 = params.smoothing_constant  # n0 prevents drastic changes in probabilities
        self._delta = params.reset_threshold  # threshold for resetting probabilities
        self._strategy_success_counts = np.zeros(self._K)
        self._strategy_probabilities = np.full(self._K, 1.0 / self._K)

        # Archive
        self._archive_size = params.archive_size
        self._archive = []

        # Population size reduction
        self._min_pop_size = params.minimum_population_size
        self._start_population_size = self.population_size
        self._population_size_reduction_strategy = params.population_reduction_strategy

        # Algorithm parameters
        self._p = params.best_member_percentage

        # Pre-computed table for converting pm to CR for exponential crossover
        # According to the paper: CR^d - d*pm*CR + d*pm - 1 = 0
        self._d_p = params.pm_to_cr_table_size  # number of intervals for pre-computed table
        self._pm_to_cr_table = self._precompute_pm_to_cr_table()

        # Crossover and mutation components
        self._index_gen = IndexGenerator()
        self._binomial_crossing = BinomialCrossover()
        self._exponential_crossing = ExponentialCrossover()
        self._archive_reduction = ArchiveReduction()
        self._random_value_gen = RandomValueGenerator()

        # Statistics
        self._strategy_usage_stats = []
        self._strategy_success_stats = []

    def _precompute_pm_to_cr_table(self) -> Dict[float, float]:
        """
        Pre-compute table mapping pm -> CR for exponential crossover.

        Solves:
            CR^d - d*pm*CR + d*pm - 1 = 0

        If there are two solutions, the smaller one is stored.
        If only CR = 1 exists, then 1 is stored.
        """
        table = {}

        d = self.nr_of_args
        pm_min = 1.0 / d
        pm_max = 1.0

        def f(cr, pm):
            return cr ** d - d * pm * cr + d * pm - 1

        for r in range(self._d_p + 1):
            pm = pm_min + r * (pm_max - pm_min) / self._d_p

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
                        # fhi = fm
                    else:
                        lo = mid
                        flo = fm

                cr = 0.5 * (lo + hi)

            table[pm] = float(np.clip(cr, 0.0, 1.0))

        return table

    def _pm_to_cr(self, pm: float) -> float:
        """
        Converts pm (mutation probability for exponential crossover) to CR.

        Uses the pre-computed table for efficiency.

        Args:
            pm: Mutation probability in [1/d, 1]

        Returns:
            Corresponding CR value.
        """
        # Clip pm to valid range
        pm = np.clip(pm, 1.0 / self.nr_of_args, 1.0)

        # Find the closest pm in the table
        closest_pm = min(self._pm_to_cr_table.keys(), key=lambda x: abs(x - pm))
        return self._pm_to_cr_table[closest_pm]

    def _select_strategy(self) -> int:
        """
        Selects a DE strategy using roulette-wheel selection.

        The selection probabilities are proportional to the success
        of each strategy in previous generations.

        Returns:
            Index of the selected strategy.
        """
        return np.random.choice(self._K, p=self._strategy_probabilities)

    def _generate_parameters(self, strategy_idx: int) -> Tuple[float, float, int, float]:
        """
        Generates control parameters F and CR for a specific strategy.

        For binomial crossover strategies, CR is generated directly.
        For exponential crossover strategies, pm is generated and converted to CR.

        Args:
            strategy_idx: Index of the strategy (0-3)

        Returns:
            f: Mutation factor
            cr: Crossover rate
            crossover_type: 0 for binomial, 1 for exponential
            pm: Mutation probability (only for exponential crossover, else None)
        """
        # Randomly select a memory index for this strategy
        ri = self._index_gen.generate(0, self._H)

        # Generate F from Cauchy distribution
        f = self._random_value_gen.generate_cauchy_greater_than_zero(
            mean=self._memory_F[strategy_idx, ri],
            scale=0.1,
            max_val=1.0
        )

        # Determine crossover type and generate appropriate parameter
        is_exponential = strategy_idx in [self.STRATEGY_CE, self.STRATEGY_RE]

        if is_exponential:
            # For exponential crossover: generate pm from normal distribution
            pm = self._random_value_gen.generate_normal(
                mean=self._memory_Cr[strategy_idx, ri],
                std_dev=0.1,
                min_val=1.0 / self.nr_of_args,
                max_val=1.0
            )
            cr = self._pm_to_cr(pm)
            crossover_type = self.CROSSOVER_EXPONENTIAL
        else:
            # For binomial crossover: generate CR from normal distribution
            cr = self._random_value_gen.generate_normal(
                mean=self._memory_Cr[strategy_idx, ri],
                std_dev=0.1,
                min_val=0.0,
                max_val=1.0
            )
            pm = None  # Not used for binomial crossover
            crossover_type = self.CROSSOVER_BINOMIAL

        return f, cr, crossover_type, pm

    def _update_memory_for_strategy(self, strategy_idx: int):
        """
        Updates the historical circle memory for a specific strategy.

        The update is based on successful individuals collected
        during the current generation for that strategy.

        M_CR is updated using a weighted arithmetic mean,
        while M_F is updated using a weighted Lehmer mean.

        Args:
            strategy_idx: Index of the strategy to update.
        """
        if len(self._successF[strategy_idx]) == 0 or len(self._successCr[strategy_idx]) == 0:
            return

        total_diff = np.sum(self._difference_fitness_success[strategy_idx])

        if total_diff <= 0:
            return

        weights = np.array(self._difference_fitness_success[strategy_idx]) / total_diff

        # Update M_CR (weighted arithmetic mean)
        cr_new = np.sum(weights * np.array(self._successCr[strategy_idx]))
        cr_new = np.clip(cr_new, 0, 1)
        self._memory_Cr[strategy_idx, self._k_indices[strategy_idx]] = cr_new

        # Update M_F (weighted Lehmer mean)
        f_values = np.array(self._successF[strategy_idx])
        f_num = np.sum(weights * f_values * f_values)
        f_den = np.sum(weights * f_values)
        if f_den > 0:
            f_new = f_num / f_den
            f_new = np.clip(f_new, 0, 1)
            self._memory_F[strategy_idx, self._k_indices[strategy_idx]] = f_new

        # Increment memory pointer for this strategy
        self._k_indices[strategy_idx] = (self._k_indices[strategy_idx] + 1) % self._H

    def _update_memories(self):
        """
        Updates the historical circle memories for all strategies that had successes.
        """
        for k in range(self._K):
            if len(self._successF[k]) > 0:
                self._update_memory_for_strategy(k)

    def _update_strategy_probabilities(self):
        """
        Updates strategy selection probabilities.

        The probability of selecting each strategy is proportional to its
        success count, with smoothing controlled by n0.

        If any probability falls below threshold delta, the entire probability
        vector is reset to a uniform distribution.
        """
        total_successes = np.sum(self._strategy_success_counts)

        if total_successes == 0:
            return

        # Update probabilities using equation (1) from the paper
        for k in range(self._K):
            self._strategy_probabilities[k] = (
                                                      self._strategy_success_counts[k] + self._n0
                                              ) / (total_successes + self._K * self._n0)

        self._strategy_probabilities = self._strategy_probabilities / np.sum(self._strategy_probabilities)

        # Reset if any probability falls below delta
        if np.any(self._strategy_probabilities < self._delta):
            self._strategy_probabilities = np.full(self._K, 1.0 / self._K)
            return

        # Store statistics
        self._strategy_usage_stats.append(copy.deepcopy(self._strategy_probabilities))
        self._strategy_success_stats.append(copy.deepcopy(self._strategy_success_counts))

    def _mutate(self, i: int, strategy_idx: int, f: float) -> np.ndarray:
        """
        Applies mutation based on the selected strategy.

        Args:
            i: Index of the current individual
            strategy_idx: Index of the selected strategy
            f: Mutation factor

        Returns:
            Mutant vector.
        """
        current_member = self._pop.members[i]

        if strategy_idx in [self.STRATEGY_CB, self.STRATEGY_CE]:
            # current-to-pbest/1 mutation
            r1 = self._index_gen.generate_unique(self._pop.size, [i])

            archive_pop = list(self._pop.members) + list(self._archive)
            archive_len = len(archive_pop)

            r2 = self._index_gen.generate_unique(archive_len, [i, r1])

            best_count = int(self._p * self._pop.size)
            best_members = self._pop.get_best_members(best_count)

            pbest_idx = self._index_gen.generate(0, len(best_members))
            pbest = best_members[pbest_idx]

            mutant = MutationCurrentToPBest1.mutate(
                base_member=current_member,
                best_member=pbest,
                r1=self._pop.members[r1],
                r2=archive_pop[r2],
                f=f
            )
        else:  # STRATEGY_RB or STRATEGY_RE
            # randrl/1 mutation (tournament best among three random points)
            mutant = MutationRandrl1.mutate(
                population=self._pop,
                current_index=i,
                f=f,
                optimization=self._pop.optimization
            )

        return mutant

    def next_epoch(self):
        """
        Perform the next epoch of the LSHADE44 algorithm.
        """
        # Reset success sets for each strategy
        for k in range(self._K):
            self._successF[k] = []
            self._successCr[k] = []
            self._difference_fitness_success[k] = []

        # Clear strategy success counts
        self._strategy_success_counts = np.zeros(self._K)

        # First pass: select strategies and generate parameters for all individuals
        f_table = np.zeros(self._pop.size)
        cr_table = np.zeros(self._pop.size)
        pm_table = np.zeros(self._pop.size)
        crossover_types = np.zeros(self._pop.size, dtype=int)
        strategy_indices = np.zeros(self._pop.size, dtype=int)

        for i in range(self._pop.size):
            strategy = self._select_strategy()
            f, cr, crossover_type, pm = self._generate_parameters(strategy)  # JESLI nie ma PM PM NONE

            strategy_indices[i] = strategy
            f_table[i] = f
            cr_table[i] = cr
            pm_table[i] = pm  # it can be None if the strategy is binomial crossover
            crossover_types[i] = crossover_type

        # Generate trial vectors
        trial_members = [None] * self._pop.size

        for i in range(self._pop.size):
            strategy_idx = strategy_indices[i]
            current_member = self._pop.members[i]

            # Mutation
            mutant = self._mutate(i, strategy_idx, f_table[i])

            # Crossover
            if crossover_types[i] == self.CROSSOVER_BINOMIAL:
                trial = self._binomial_crossing.crossover_members(
                    current_member, mutant, cr_table[i]
                )
            else:
                trial = self._exponential_crossing.crossover_members(
                    current_member, mutant, cr_table[i]
                )

            trial_members[i] = trial

        trial_pop = Population.with_new_members(self._pop, trial_members)

        # Fix boundary constraints
        fix_boundary_constraints_with_parent(self._pop, trial_pop, self.boundary_constraints_fun)

        # Evaluate trial population
        trial_pop.update_fitness_values(self._function.eval, self.parallel_processing)

        # Selection with strategy performance tracking
        new_members = []

        for i in range(self._pop.size):
            strategy = strategy_indices[i]
            origin_member = self._pop.members[i]
            trial_member = trial_pop.members[i]

            is_better = False
            if self._pop.optimization == OptimizationType.MINIMIZATION:
                is_better = trial_member.fitness_value < origin_member.fitness_value
            else:
                is_better = trial_member.fitness_value > origin_member.fitness_value

            if is_better:
                # Count success for this strategy
                self._strategy_success_counts[strategy] += 1

                # Store successful parameters for this strategy
                self._successF[strategy].append(f_table[i])

                if crossover_types[i] == self.CROSSOVER_EXPONENTIAL:
                    self._successCr[strategy].append(pm_table[i])  # Save pm for exponential crossover
                else:
                    self._successCr[strategy].append(cr_table[i])  # Save cr for binomial crossover

                self._difference_fitness_success[strategy].append(
                    abs(origin_member.fitness_value - trial_member.fitness_value)
                )

                # Add to archive
                self._archive.append(copy.deepcopy(origin_member))

                # Keep trial member
                new_members.append(copy.deepcopy(trial_member))
            else:
                # Keep original member
                new_members.append(copy.deepcopy(origin_member))

        # Update population
        self._pop = Population.with_new_members(self._pop, new_members)

        # Reduce archive if needed
        self._archive = self._archive_reduction.reduce_archive(
            self._archive, self._archive_size, self.population_size
        )

        # Update historical circle memories for each strategy
        self._update_memories()

        # Update strategy probabilities
        self._update_strategy_probabilities()

        # Apply linear population size reduction
        self.update_population_size(self.nfe, self.nfe_max, self._start_population_size,
                                    self._min_pop_size)

    def update_population_size(self, nfe: int, total_nfe: int, start_pop_size: int, min_pop_size: int):
        """
        Calculate new population size using Linear Population Size Reduction (LPSR).

        According to equation (13) in the paper:
        N = round[N_init - (FES / MaxFES) * (N_init - N_min)]
        """
        new_size = self._population_size_reduction_strategy.get_new_population_size(
            nfe, total_nfe, start_pop_size, min_pop_size
        )
        self._pop.resize(new_size)

    def get_strategy_statistics(self) -> dict:
        """
        Compute and return statistics describing the usage and performance of all DE strategies.
        """
        if not self._strategy_usage_stats:
            return {
                'probabilities': [],
                'success_counts': [],
                'average_probabilities': np.zeros(self._K),
                'total_successes': np.zeros(self._K),
                'usage_percentage': np.zeros(self._K),
                'final_probabilities': self._strategy_probabilities.copy(),
                'total_epochs': 0,
                'memory_F': self._memory_F.tolist(),
                'memory_Cr': self._memory_Cr.tolist()
            }

        probs_array = np.array(self._strategy_usage_stats)
        success_array = np.array(self._strategy_success_stats)

        avg_probs = np.mean(probs_array, axis=0)
        total_successes = np.sum(success_array, axis=0)
        total_epochs = len(self._strategy_usage_stats)
        usage_percentage = avg_probs * 100

        return {
            'probabilities': self._strategy_usage_stats,
            'success_counts': self._strategy_success_stats,
            'average_probabilities': avg_probs,
            'total_successes': total_successes,
            'usage_percentage': usage_percentage,
            'final_probabilities': self._strategy_probabilities.copy(),
            'total_epochs': total_epochs,
            'memory_F': self._memory_F.tolist(),
            'memory_Cr': self._memory_Cr.tolist()
        }

    def print_strategy_statistics(self):
        """
        Print detailed statistics about the usage and performance of all competing DE strategies.
        """
        stats = self.get_strategy_statistics()

        print("\n" + "=" * 80)
        print("STRATEGY STATISTICS FOR LSHADE44")
        print("=" * 80)

        print("\n{:<25} {:>12} {:>15} {:>15} {:>12}".format(
            "Strategy", "Avg Prob %", "Total Success", "Usage %", "Final Prob %"
        ))
        print("-" * 80)

        for k in range(self._K):
            name = self.STRATEGY_NAMES[k]
            avg_prob = stats['average_probabilities'][k] * 100
            total_succ = stats['total_successes'][k]
            usage_pct = stats['usage_percentage'][k]
            final_prob = stats['final_probabilities'][k] * 100

            print("{:<25} {:>12.2f} {:>15.0f} {:>15.2f} {:>12.2f}".format(
                name, avg_prob, total_succ, usage_pct, final_prob
            ))

        print("-" * 80)
        print(f"Total epochs: {stats['total_epochs']}")
        print(f"Total successes: {np.sum(stats['total_successes'])}")

        # Print memory information
        print("\nMEMORY STATE PER STRATEGY:")
        print("-" * 80)
        for k in range(self._K):
            print(f"{self.STRATEGY_NAMES[k]}:")
            print(f"  MF: {stats['memory_F'][k][:5]}... (first 5 values)")
            print(f"  MC: {stats['memory_Cr'][k][:5]}... (first 5 values)")
            print(f"  Pointer: {self._k_indices[k]}")

        self._print_probability_evolution(stats)

    def _print_probability_evolution(self, stats: dict, max_points: int = 20):
        """
        Display the evolution of strategy selection probabilities over time.
        """
        probs = stats['probabilities']
        if not probs:
            return

        print("\nPROBABILITY EVOLUTION (sampled every {} epochs):".format(
            max(1, len(probs) // max_points)
        ))
        print("-" * 80)

        step = max(1, len(probs) // max_points)
        sampled_indices = list(range(0, len(probs), step))

        header = "Epoch"
        for k in range(self._K):
            header += f"  {self.STRATEGY_SHORT_NAMES[k]:>6}"
        print(header)
        print("-" * 80)

        for idx in sampled_indices[:max_points]:
            row = f"{idx:>5}"
            for k in range(self._K):
                row += f"  {probs[idx][k] * 100:>6.1f}"
            print(row)

        if sampled_indices[-1] != len(probs) - 1:
            idx = len(probs) - 1
            row = f"{idx:>5}"
            for k in range(self._K):
                row += f"  {probs[idx][k] * 100:>6.1f}"
            print(row)

        print("-" * 80)

    def get_detailed_strategy_report(self) -> dict:
        """
        Generate a detailed structured report describing the behavior of all DE strategies.
        """
        stats = self.get_strategy_statistics()

        report = {}
        for k in range(self._K):
            report[self.STRATEGY_NAMES[k]] = {
                'short_name': self.STRATEGY_SHORT_NAMES[k],
                'average_probability': stats['average_probabilities'][k],
                'total_successes': int(stats['total_successes'][k]),
                'usage_percentage': stats['usage_percentage'][k],
                'final_probability': stats['final_probabilities'][k],
                'memory_F': stats['memory_F'][k],
                'memory_Cr': stats['memory_Cr'][k],
                'memory_pointer': int(self._k_indices[k])
            }

        report['total_epochs'] = stats['total_epochs']
        report['total_successes'] = int(np.sum(stats['total_successes']))
        report['final_probabilities'] = stats['final_probabilities'].tolist()

        if stats['total_epochs'] > 0 and np.sum(stats['average_probabilities']) > 0:
            report['dominant_strategy'] = self.STRATEGY_NAMES[np.argmax(stats['average_probabilities'])]
        else:
            report['dominant_strategy'] = 'N/A (no data)'

        return report
