import copy
import numpy as np
from typing import List

from detpy.DETAlgs.base import BaseAlg
from detpy.DETAlgs.data.alg_data import ALSHADEData
from detpy.DETAlgs.methods.methods_alshade import current_to_xamean
from detpy.DETAlgs.methods.methods_lshade import calculate_best_member_count, crossing, archive_reduction
from detpy.DETAlgs.mutation_methods.current_to_pbest_1 import MutationCurrentToPBest1
from detpy.models.enums.boundary_constrain import fix_boundary_constraints_with_parent
from detpy.models.enums.optimization import OptimizationType
from detpy.models.population import Population


class ALSHADE(BaseAlg):
    """
        ALSHADE: A novel adaptive L-SHADE algorithm and its application in UAV swarm resource configuration problem

        References:
        Yintong Li, Tong Han, Huan Zhou, Shangqin Tang, Hui Zhao Information Sciences, vol. 606, pp. 350–367, 2022.
    """

    def __init__(self, params: ALSHADEData, db_conn=None, db_auto_write=False):
        super().__init__(ALSHADE.__name__, params, db_conn, db_auto_write)

        self.H = params.memory_size
        self.memory_F = np.full(self.H, 0.5)
        self.memory_Cr = np.full(self.H, 0.5)

        self.memory_Cr[0] = 0.9
        self.memory_F[0] = 0.9
        self.index_update_f_and_cr = 1

        self.successCr = []
        self.successF = []
        self.difference_fitness_success = []

        self.e = params.elite_factor
        self.P = params.init_probability_mutation_strategy

        self.archive = []
        self.archive_size = params.archive_size

        self.archive.append(copy.deepcopy(self._pop.get_best_members(1)[0]))

        self.min_pop_size = params.minimum_population_size
        self.start_population_size = self.population_size

        self._mutation_memory = []

        self.population_size_reduction_strategy = params.population_size_reduction_strategy

        self.nfe_max = self.nfe_max

    def update_population_size(self, nfe: int, total_nfe: int, start_pop_size: int, min_pop_size: int):
        """
        Update the population size based on the current epoch using the specified population size reduction strategy.

        Parameters:
        - nfe (int): The current nfe number.
        - total_nfe (int): The total number of function evaluations.
        - start_pop_size (int): The initial population size.
        - min_pop_size (int): The minimum population size.

        """
        new_size = self.population_size_reduction_strategy.get_new_population_size(
           nfe, total_nfe, start_pop_size, min_pop_size
        )
        self._pop.resize(new_size)
        self.archive_size = new_size

    def _compute_weighted_archive_mean(self):
        """
        Compute the weighted mean from the archive.

        """
        m = round(self.e * len(self.archive))
        if m == 0:
            return np.mean([[chrom.real_value for chrom in ind.chromosomes] for ind in self.archive], axis=0)

        reverse_sort = (self._pop.optimization == OptimizationType.MAXIMIZATION)
        sorted_archive = sorted(self.archive, key=lambda ind: ind.fitness_value, reverse=reverse_sort)

        top_m = sorted_archive[:m]
        weights = np.log(m + 0.5) - np.log(np.arange(1, m + 1))
        weights /= np.sum(weights)

        return np.sum([
            [w * chrom.real_value for chrom in ind.chromosomes]
            for w, ind in zip(weights, top_m)
        ], axis=0)

    def mutate(self, population: Population, the_best_to_select_table: List[int], f_table: List[float]) -> Population:
        """
        The mutation method of the ALSHADE algorithm.

        Parameters:
        - population (Population): The current population.
        - the_best_to_select_table (List[int]): A list where each element specifies the number of best members to select for the mutation process for the corresponding individual in the population.
        - f_table (List[float]): A list of scaling factors (F) for each individual in the population.

        """
        new_members = []

        # Contains 1 if "current-to-pbest/1" was used, 3 if "current to xamean" was used
        memory = []

        pa = np.concatenate((population.members, self.archive))

        for i in range(population.size):
            r1 = np.random.choice(len(population.members))
            r2 = np.random.choice(len(pa))
            p = np.random.rand()

            if p < self.P:
                best_members = population.get_best_members(the_best_to_select_table[i])
                best = best_members[np.random.randint(0, len(best_members))]

                mutant = MutationCurrentToPBest1.mutate(
                    base_member=population.members[i],
                    best_member=best,
                    r1=population.members[r1],
                    r2=pa[r2],
                    f=f_table[i]
                )

                memory.append(1)
            else:
                xamean = self._compute_weighted_archive_mean()

                mutant = current_to_xamean(
                    base_member=population.members[i],
                    xamean=xamean,
                    r1=population.members[r1],
                    r2=pa[r2],
                    f=f_table[i]
                )
                # current to xamean
                memory.append(3)
            new_members.append(mutant)

        new_population = Population(
            lb=population.lb, ub=population.ub,
            arg_num=population.arg_num, size=population.size,
            optimization=population.optimization
        )
        new_population.members = np.array(new_members)
        self._mutation_memory = memory
        return new_population

    def selection(self, origin: Population, modified: Population, ftable: List[float], cr_table: List[float]):
        """
        The selection method of the ALSHADE algorithm.

        Parameters:
        - population (Population): The current population.
        - modified (Population): The population after mutation and crossover.
        - f_table (List[float]): A list of scaling factors (F) for each individual in the population.
        - cr_table (List[float]): A list of crossover rates (CR) for each individual in the population.

        """
        new_members = []
        memory_flags = self._mutation_memory
        for i in range(origin.size):
            if (origin.optimization == OptimizationType.MINIMIZATION and origin.members[i] <= modified.members[i]) or \
                    (origin.optimization == OptimizationType.MAXIMIZATION and origin.members[i] >= modified.members[i]):
                new_members.append(copy.deepcopy(origin.members[i]))
            else:
                self.archive.append(copy.deepcopy(origin.members[i]))
                self.successF.append(ftable[i])
                self.successCr.append(cr_table[i])
                df = abs(modified.members[i].fitness_value - origin.members[i].fitness_value)
                self.difference_fitness_success.append(df)
                new_members.append(copy.deepcopy(modified.members[i]))

        a1_all = sum(1 for m in memory_flags if m == 1)
        a1_better = sum(1 for m, f in zip(memory_flags, new_members) if m == 1 and f in modified.members)
        a2_all = sum(1 for m in memory_flags if m == 3)
        a2_better = sum(1 for m, f in zip(memory_flags, new_members) if m == 3 and f in modified.members)
        if a1_all > 0 and a2_all > 0:
            p_a1 = a1_better / a1_all
            p_a2 = a2_better / a2_all
            self.P += 0.05 * (1 - self.P) * (p_a1 - p_a2) * self.nfe / self.nfe_max
            self.P = min(0.9, max(0.1, self.P))

        new_population = Population(
            lb=origin.lb, ub=origin.ub,
            arg_num=origin.arg_num, size=origin.size,
            optimization=origin.optimization
        )
        new_population.members = np.array(new_members)
        return new_population

    def update_memory(self, success_f: List[float], success_cr: List[float], df):
        """
        The method to update the memory of scaling factors (F) and crossover rates (CR) in the ALSHADE algorithm.

        Parameters:
        - success_f (List[float]): A list of successful scaling factors F that resulted in better fitness function values from the current epoch.
        - success_cr (List[float]): A list of successful scaling factors CR that resulted in better fitness function values from the current epoch.
        - df (List[float]): A list of differences between old and new value fitness function

        """
        if not success_f:
            return
        total = sum(df)
        weights = np.array(df) / total
        f_new = np.sum(weights * np.square(success_f)) / np.sum(weights * np.array(success_f))
        cr_new = np.sum(weights * success_cr)
        f_new = np.clip(f_new, 0, 1)
        cr_new = np.clip(cr_new, 0, 1)

        self.memory_F[self.index_update_f_and_cr] = f_new
        self.memory_Cr[self.index_update_f_and_cr] = cr_new
        self.index_update_f_and_cr += 1
        if self.index_update_f_and_cr >= len(self.memory_F):
            self.index_update_f_and_cr = 1

        self.successF = []
        self.successCr = []
        self.difference_fitness_success = []

    def initialize_parameters_for_epoch(self):
        """
        Method to create f_table, cr_table and bests_to_select for the epoch.

        """
        f_table = []
        cr_table = []
        bests_to_select = []
        for _ in range(self._pop.size):
            ri = np.random.randint(0, self.H)
            cr = np.clip(np.random.normal(self.memory_Cr[ri], 0.1), 0.0, 1.0)
            while True:
                f = np.random.standard_cauchy() * 0.1 + self.memory_F[ri]
                if f > 0:
                    break
            f = min(f, 1.0)
            f_table.append(f)
            cr_table.append(cr)
            bests_to_select.append(calculate_best_member_count(self.population_size))
        return f_table, cr_table, bests_to_select

    def next_epoch(self):
        f_table, cr_table, bests_to_select = self.initialize_parameters_for_epoch()

        mutant = self.mutate(self._pop, bests_to_select, f_table)

        trial = crossing(self._pop, mutant, cr_table)
        fix_boundary_constraints_with_parent(self._pop, trial, self.boundary_constraints_fun)
        trial.update_fitness_values(self._function.eval, self.parallel_processing)

        self._pop = self.selection(self._pop, trial, f_table, cr_table)
        self.archive = archive_reduction(self.archive, self.archive_size, self.population_size)
        self.update_memory(self.successF, self.successCr, self.difference_fitness_success)
        self.update_population_size(self.nfe, self.nfe_max, self.start_population_size,
                                    self.min_pop_size)
