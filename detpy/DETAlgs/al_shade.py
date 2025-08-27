import copy
import numpy as np
from typing import List

from detpy.DETAlgs.base import BaseAlg
from detpy.DETAlgs.data.alg_data import ALSHADEData
from detpy.DETAlgs.methods.methods_alshade import mutation_internal, current_to_xamean, fix_boundary_constraints
from detpy.DETAlgs.methods.methods_lshade import calculate_best_member_count, crossing, archive_reduction
from detpy.models.enums.optimization import OptimizationType
from detpy.models.population import Population


class ALSHADE(BaseAlg):
    def __init__(self, params: ALSHADEData, db_conn=None, db_auto_write=False):
        super().__init__(ALSHADE.__name__, params, db_conn, db_auto_write)

        self.H = params.memory_size
        self.memory_F = np.full(self.H, 0.5)
        self.memory_Cr = np.full(self.H, 0.5)

        self.successCr = []
        self.successF = []
        self.difference_fitness_success = []

        self.e = params.elite_factor
        self.P = params.init_probability_mutation_strategy

        self.archive = []
        rarc = 2.6
        self.archive_size = round(rarc * self.population_size)

        self.archive.append(copy.deepcopy(self._pop.get_best_members(1)[0]))
        self.nfe = 0
        self.min_pop_size = params.minimum_population_size
        self.start_population_size = self.population_size
        self.nfe_max = self.calculate_max_evaluations_lpsr(self.population_size)

        self._mutation_memory = []

    def calculate_max_evaluations_lpsr(self, start_pop_size):
        total_evaluations = 0
        for generation in range(self.num_of_epochs):
            current_population_size = int(
                start_pop_size - (generation / self.num_of_epochs) * (start_pop_size - self.min_pop_size)
            )
            total_evaluations += current_population_size
        return total_evaluations

    def update_population_size(self, start_pop_size, epoch, total_epochs, min_size):
        new_size = int(start_pop_size - (epoch / total_epochs) * (start_pop_size - min_size))
        self._pop.resize(new_size)
        self.archive_size = new_size

    def compute_weighted_archive_mean(self):
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
        new_members = []
        memory = []

        pa = np.concatenate((population.members, self.archive))

        for i in range(population.size):
            r1 = np.random.choice(len(population.members))
            r2 = np.random.choice(len(pa))
            p = np.random.rand()

            if p < self.P:
                best_members = population.get_best_members(the_best_to_select_table[i])
                best = best_members[np.random.randint(0, len(best_members))]

                mutant = mutation_internal(
                    base_member=population.members[i],
                    best_member=best,
                    r1=population.members[r1],
                    r2=pa[r2],
                    f=f_table[i]
                )
                memory.append(1)
            else:
                xamean = self.compute_weighted_archive_mean()

                mutant = current_to_xamean(
                    base_member=population.members[i],
                    xamean=xamean,
                    r1=population.members[r1],
                    r2=pa[r2],
                    f=f_table[i]
                )
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

    def selection(self, origin, modified, ftable, cr_table):
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

    def update_memory(self, success_f, success_cr, df):
        if not success_f:
            return
        total = sum(df)
        weights = np.array(df) / total
        f_new = np.sum(weights * np.square(success_f)) / np.sum(weights * np.array(success_f))
        cr_new = np.sum(weights * success_cr)
        f_new = np.clip(f_new, 0, 1)
        cr_new = np.clip(cr_new, 0, 1)
        self.memory_F = np.append(self.memory_F[1:], f_new)
        self.memory_Cr = np.append(self.memory_Cr[1:], cr_new)
        self.successF = []
        self.successCr = []
        self.difference_fitness_success = []

    def initialize_parameters_for_epoch(self):
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
        fix_boundary_constraints(self._pop, trial)
        trial.update_fitness_values(self._function.eval, self.parallel_processing)
        self.nfe += self._pop.size
        self._pop = self.selection(self._pop, trial, f_table, cr_table)
        self.archive = archive_reduction(self.archive, self.archive_size, self.population_size)
        self.update_memory(self.successF, self.successCr, self.difference_fitness_success)
        self.update_population_size(self.start_population_size, self._epoch_number, self.num_of_epochs,
                                    self.min_pop_size)
        self._epoch_number += 1
