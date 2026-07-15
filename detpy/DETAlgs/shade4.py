import copy
import numpy as np
from typing import List, Tuple

from detpy.DETAlgs.archive_reduction.archive_reduction import ArchiveReduction
from detpy.DETAlgs.base import BaseAlg
from detpy.DETAlgs.crossover_methods.binomial_crossover import BinomialCrossover
from detpy.DETAlgs.crossover_methods.exponential_crossover import ExponentialCrossover
from detpy.DETAlgs.data.alg_data import SHADE4Data
from detpy.DETAlgs.mutation_methods.current_to_pbest_1 import MutationCurrentToPBest1
from detpy.DETAlgs.mutation_methods.mutation_randrl_1 import MutationRandrl1
from detpy.DETAlgs.random.index_generator import IndexGenerator
from detpy.DETAlgs.random.random_value_generator import RandomValueGenerator
from detpy.models.enums.boundary_constrain import fix_boundary_constraints_with_parent
from detpy.models.enums.optimization import OptimizationType
from detpy.models.population import Population


class SHADE4(BaseAlg):
    """
    SHADE4: Success-History Based Adaptive Differential Evolution with Competing Strategies

    References: Petr Bujok, Josef Tvrdık, Radka Polakova Department of Computer Science
    University of Ostrava,
    """

    # Strategy
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

    def __init__(self, params: SHADE4Data, db_conn=None, db_auto_write=False):
        super().__init__(SHADE4.__name__, params, db_conn, db_auto_write)

        MIN_POP_SIZE_FOR_PBEST = 10
        if self.population_size < MIN_POP_SIZE_FOR_PBEST:
            raise ValueError(
                "SHADE requires population_size >= 10 "
                "because p is sampled from [2/NP, 0.2]."
            )

        self._H = params.memory_size  # Memory size for f and cr adaptation
        self._archive_size = self.population_size  # Size of the archive is the same as population
        self._archive = []  # Archive for storing the members from old populations

        self._k_index = 0  # Index for memory update
        self._memory_F = np.full(self._H, 0.5)  # Initial memory for F
        self._memory_Cr = np.full(self._H, 0.5)  # Initial memory for CR

        self._K = 4  # Number of strategies
        self._n0 = params.smoothing_constant  # n0 introduced to prevent drastic changes in strategy probabilities q_k
        # when all strategy success frequencies n_k are small or zero.

        self._delta = params.reset_threshold  # delta threshold for resetting strategy probabilities;
        # if any probabilities falls below delta, all strategy probabilities are reset to a uniform distribution

        self._strategy_success_counts = np.zeros(self._K)
        self._strategy_probabilities = np.full(self._K, 1.0 / self._K)

        self._successF = []
        self._successCr = []
        self._difference_fitness_success = []

        # Statistic strategy
        self._strategy_usage_stats = []
        self._strategy_success_stats = []

        self._index_gen = IndexGenerator()
        self._binomial_crossing = BinomialCrossover()
        self._exponential_crossing = ExponentialCrossover()
        self._archive_reduction = ArchiveReduction()
        self._random_value_gen = RandomValueGenerator()

    def _select_strategy(self) -> int:
        """
        Selects a DE strategy using roulette-wheel selection.

        The selection probabilities are proportional to the success
        of each strategy in previous generations.

        Returns:
            Index of the selected strategy.
        """
        return np.random.choice(self._K, p=self._strategy_probabilities)

    def _generate_parameters(self) -> Tuple[float, float, int]:
        """
        Generates control parameters F, CR, and pbest size for SHADE.

        F is sampled from a Cauchy distribution centered at a randomly
        selected historical memory value M_F.

        CR is sampled from a normal distribution centered at a randomly
        selected historical memory value M_CR and clipped to [0, 1].

        The pbest proportion is sampled uniformly from [2/NP, 0.2] and
        converted into an integer number of top individuals used in
        current-to-pbest mutation.

        Returns:
            f (float): mutation factor
            cr (float): crossover rate
            best_count (int): number of pbest candidates
        """
        ri = self._index_gen.generate(0, self._H)

        cr = self._random_value_gen.generate_normal(
            mean=self._memory_Cr[ri],
            std_dev=0.1,
            min_val=0.0,
            max_val=1.0
        )

        f = self._random_value_gen.generate_cauchy_greater_than_zero(
            mean=self._memory_F[ri],
            scale=0.1,
            max_val=1.0
        )

        best_count = self._random_value_gen.generate_count_from_percentage(
            total_size=self.population_size,
            min_percentage=2.0 / self.population_size,
            max_percentage=0.2
        )
        best_count = max(1, best_count)

        return f, cr, best_count

    def _update_memory(self):
        """
        Updates the shared success-history memory M_F and M_CR.

        The update is based on successful individuals collected
        during the current generation.

        Each successful parameter set is weighted proportionally to the
        fitness improvement it produced, ensuring that better improvements
        have a stronger influence on the memory update.

        M_CR is updated using a weighted arithmetic mean,
        while M_F is updated using a weighted Lehmer mean.

        The memory index k is updated in a cyclic manner.
        """
        if len(self._successF) == 0 or len(self._successCr) == 0:
            return

        total_diff = np.sum(self._difference_fitness_success)

        if total_diff <= 0:
            return

        weights = np.array(self._difference_fitness_success) / total_diff

        cr_new = np.sum(weights * np.array(self._successCr))
        cr_new = np.clip(cr_new, 0, 1)
        self._memory_Cr[self._k_index] = cr_new

        f_num = np.sum(weights * np.array(self._successF) * np.array(self._successF))
        f_den = np.sum(weights * np.array(self._successF))
        if f_den > 0:
            f_new = f_num / f_den
            f_new = np.clip(f_new, 0, 1)
            self._memory_F[self._k_index] = f_new

        self._k_index = (self._k_index + 1) % self._H

    def _update_strategy_probabilities(self):
        """
        Updates strategy selection probabilities.

        The probability of selecting each strategy is proportional to its
        success count, with smoothing controlled by n0 to avoid abrupt changes
        when success frequencies are low.

        If any probability falls below threshold sigma, the entire probability
        vector is reset to a uniform distribution to maintain exploration.

        The resulting probabilities are normalized and stored for statistical
        analysis of strategy usage over time.
        """


        total_successes = np.sum(self._strategy_success_counts)

        if total_successes == 0:
            return

        for k in range(self._K):
            self._strategy_probabilities[k] = (
                                                      self._strategy_success_counts[k] + self._n0
                                              ) / (total_successes + self._K * self._n0)

        self._strategy_probabilities = self._strategy_probabilities / np.sum(self._strategy_probabilities)

        if np.any(self._strategy_probabilities < self._delta):
            self._strategy_probabilities = np.full(self._K, 1.0 / self._K)
            return

        self._strategy_usage_stats.append(copy.deepcopy(self._strategy_probabilities))
        self._strategy_success_stats.append(copy.deepcopy(self._strategy_success_counts))

    def next_epoch(self):
        """
        Perform the next epoch of the SHADE4 algorithm.
        """
        f_table = []
        cr_table = []
        best_counts = []
        strategy_indices = []

        self._successF = []
        self._successCr = []
        self._difference_fitness_success = []

        for i in range(self._pop.size):
            strategy = self._select_strategy()
            f, cr, best = self._generate_parameters()

            strategy_indices.append(strategy)
            f_table.append(f)
            cr_table.append(cr)
            best_counts.append(best)


        trial_members = [None] * self._pop.size

        for i in range(self._pop.size):
            strategy_idx = strategy_indices[i]
            current_member = self._pop.members[i]

            if strategy_idx in [self.STRATEGY_CB, self.STRATEGY_CE]:
                r1 = self._index_gen.generate_unique(self._pop.size, [i])

                archive_pop = list(self._pop.members) + list(self._archive)
                archive_len = len(archive_pop)

                r2 = self._index_gen.generate_unique(archive_len, [i, r1])

                best_members = self._pop.get_best_members(best_counts[i])

                pbest_idx = self._index_gen.generate(0, len(best_members))
                pbest = best_members[pbest_idx]

                mutant = MutationCurrentToPBest1.mutate(
                    base_member=current_member,
                    best_member=pbest,
                    r1=self._pop.members[r1],
                    r2=archive_pop[r2],
                    f=f_table[i]
                )
            else:  # STRATEGY_RB or STRATEGY_RE
                mutant = MutationRandrl1.mutate(
                    population=self._pop,
                    current_index=i,
                    f=f_table[i],
                    optimization=self._pop.optimization
                )

            if strategy_idx in [self.STRATEGY_CB, self.STRATEGY_RB]:
                trial = self._binomial_crossing.crossover_members(
                    current_member, mutant, cr_table[i]
                )
            else:
                trial = self._exponential_crossing.crossover_members(
                    current_member, mutant, cr_table[i]
                )

            trial_members[i] = trial

        trial_pop = Population.with_new_members(self._pop, trial_members)

        fix_boundary_constraints_with_parent(self._pop, trial_pop, self.boundary_constraints_fun)

        trial_pop.update_fitness_values(self._function.eval, self.parallel_processing)

        new_pop = self._selection(self._pop, trial_pop, f_table, cr_table, strategy_indices)

        self._archive = self._archive_reduction.reduce_archive(
            self._archive, self._archive_size, self.population_size
        )

        self._pop = new_pop

        self._update_memory()

        self._update_strategy_probabilities()

    def _selection(self, origin: Population, trial: Population,
                   f_table: List[float], cr_table: List[float],
                   strategy_indices: List[int]) -> Population:
        """
        Selection step with strategy performance tracking.
        """
        new_members = []
        self._strategy_success_counts = np.zeros(self._K)

        pop_size = min(len(origin.members), len(trial.members))

        for i in range(pop_size):
            strategy = strategy_indices[i]
            origin_member = origin.members[i]
            trial_member = trial.members[i]

            is_better = False
            if self._pop.optimization == OptimizationType.MINIMIZATION:
                is_better = trial_member.fitness_value < origin_member.fitness_value
            else:
                is_better = trial_member.fitness_value > origin_member.fitness_value

            if is_better:
                self._strategy_success_counts[strategy] += 1
                self._archive.append(copy.deepcopy(origin_member))
                self._successF.append(f_table[i])
                self._successCr.append(cr_table[i])
                self._difference_fitness_success.append(
                    abs(origin_member.fitness_value - trial_member.fitness_value)
                )
                new_members.append(copy.deepcopy(trial_member))
            else:
                new_members.append(copy.deepcopy(origin_member))

        if not new_members:
            new_members = copy.deepcopy(origin.members)

        return Population.with_new_members(origin, new_members)

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
                'final_probabilities': np.full(self._K, 1.0 / self._K),
                'total_epochs': 0
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
            'final_probabilities': self._strategy_probabilities,
            'total_epochs': total_epochs
        }

    def print_strategy_statistics(self):
        """
        Print detailed statistics about the usage and performance of all competing DE strategies.
        """
        stats = self.get_strategy_statistics()

        print("\n" + "=" * 80)
        print("STRATEGY STATISTICS FOR SHADE4")
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
                'final_probability': stats['final_probabilities'][k]
            }

        report['total_epochs'] = stats['total_epochs']
        report['total_successes'] = int(np.sum(stats['total_successes']))
        report['shared_memory_F'] = self._memory_F.tolist()
        report['shared_memory_Cr'] = self._memory_Cr.tolist()

        if stats['total_epochs'] > 0 and np.sum(stats['average_probabilities']) > 0:
            report['dominant_strategy'] = self.STRATEGY_NAMES[np.argmax(stats['average_probabilities'])]
        else:
            report['dominant_strategy'] = 'N/A (no data)'

        return report

