import copy
from typing import List

import numpy as np
from detpy.DETAlgs.base import BaseAlg
from detpy.DETAlgs.data.alg_data import LSHADERSPData

from random import randint

from detpy.DETAlgs.methods.methods_lshadersp import crossing, archive_reduction, rank_selection
from detpy.DETAlgs.mutation_methods.current_to_pbest_r import MutationCurrentToPBestR
from detpy.math_functions.lehmer_mean import LehmerMean
from detpy.models.enums.boundary_constrain import fix_boundary_constraints_with_parent
from detpy.models.enums.optimization import OptimizationType
from detpy.models.member import Member
from detpy.models.population import Population


class LSHADERSP(BaseAlg):
    """
        L-SHADE-RSP: LSHADE Algorithm with a Rank-based Selective Pressure Strategy (RSP)

        References:
        Shakhnaz Akhmedova, Vladimir Stanovov and Eugene Semenkin (2018). Research on In Proceedings of
        the 15th International Conference on Informatics in Control, Automation and Robotics (ICINCO 2018) -
        Volume 1, pages 149-155
    """

    def __init__(self, params: LSHADERSPData, db_conn=None, db_auto_write=False):
        super().__init__(LSHADERSP.__name__, params, db_conn, db_auto_write)
        self._k = params.scaling_factor_for_rank_selection  # Scaling factor for rank selection
        self._H = params.memory_size  # Memory size for f and cr adaptation
        self._memory_F = np.full(self._H, 0.3)  # Initial memory for F
        self._memory_Cr = np.full(self._H, 0.8)  # Initial memory for Cr

        self._memory_F[self._H - 1] = 0.9  # One cell of the memory for F must be set to 0.9
        self._memory_Cr[self._H - 1] = 0.9  # One cell of the memory for Cr must be set to 0.9

        self._population_size_reduction_strategy = params.population_reduction_strategy

        self._min_pop_size = params.minimum_population_size  # Minimal population size

        self._start_population_size = self.population_size

        self._archive_size = self.population_size  # Size of the archive
        self._archive = []  # Archive for storing the members from old populations

        self._lehmer_mean_func = LehmerMean()  # Class for Lehmer mean calculation

        self._success_f = []  # List of successful F values
        self._success_cr = []  # List of successful Cr values

        self._f = np.random.standard_cauchy() * 0.1 + np.random.choice(self._memory_F)
        self._fw = 0.7 * self._f if self.nfe < 0.2 * self.nfe_max else (
            0.8 * self._f if self.nfe < 0.4 * self.nfe_max else 1.2 * self._f)

        self._difference_fitness_success = []

        # Define a terminal value for memory_Cr. It indicates that no successful Cr was found in the epoch.
        self._TERMINAL = np.nan

        # We need this value for checking close to zero in update_memory
        self._EPSILON = 0.00001

    def _adapt_parameters(self, fitness_improvement: List[float]):
        """
        Update the memory for F and Cr based on the fitness improvement of the crossover and mutation steps.

        Parameters:
        - fitness_improvement: List of fitness improvement values for successful individuals.
        """
        epsilon = 1e-10  # Small value to avoid division by zero

        total_improvement = np.sum(fitness_improvement)
        if total_improvement > epsilon:
            # Compute weights based on fitness improvement
            weights = fitness_improvement / total_improvement

            # Randomly select a memory index to update
            # Ensure one cell in the memory always holds the fixed value 0.9 so we have index -1
            r = np.random.randint(0, self._H - 1)

            # Compute new F value as the Lehmer mean of successful F values
            if len(self._success_f) > 0:
                new_f = self._lehmer_mean_func.evaluate(self._success_f, weights, 2)
                new_f = np.clip(new_f, 0, 1)  # Clip to [0, 1]
                # Update memory_F
                self._memory_F[r] = (self._memory_F[r] + new_f) / 2  # Mean of old and new F

            if np.isclose(total_improvement, 0.0, atol=self._EPSILON):
                self._memory_Cr[r] = self._TERMINAL

            else:
                new_cr = np.sum(weights * self._success_cr)
                new_cr = np.clip(new_cr, 0, 1)  # Clip to [0, 1]
                # Update memory_Cr
                self._memory_Cr[r] = (self._memory_Cr[r] + new_cr) / 2  # Mean of old and new Cr

        # Clear the lists of successful F and Cr values for the next generation
        self._difference_fitness_success = []
        self._success_cr = []
        self._success_f = []

    def _get_best_members(self, best_members: List[Member], archive: List[Member]) -> List[Member]:
        """
        Combine and sort the best members from the current population and the archive,
        then return the top members.

        Parameters:
        - best_members (List[Member]): The best members from the current population.
        - archive (List[Member]): The archive of old population members.

        Returns:
        - List[Member]: The top combined members from best_members and archive.
        """
        size_best_members_to_select = len(best_members)

        combined = np.concatenate((best_members, archive))

        if self._pop.optimization == OptimizationType.MINIMIZATION:
            # For minimization, sort the combined array by the smallest fitness value
            combined = sorted(combined, key=lambda member: member.fitness_value)
        else:
            # For maximization, sort the combined array by the largest fitness value
            combined = sorted(combined, key=lambda member: member.fitness_value, reverse=True)

        return combined[:size_best_members_to_select]

    def _mutate(self,
                population: Population,
                nfe: int,
                nfe_max: int,
                pop_size: int,
                f_table: List[float],
                fw_table: List[float]

                ) -> Population:
        """
        Perform mutation step for the population.

        Parameters:
        - population (Population): The population to mutate.
        - nfe (int): The current number of function evaluations.
        - nfe_max (int): The maximum number of function evaluations.
        - pop_size (int): The size of the population.
        - f_table (List[float]): A list of scaling factors F for the current epoch.
        - fw_table (List[float]): A list of dynamic scaling factors Fw for the current epoch.

        Returns: A new Population with mutated members.
        """

        new_members = []
        for i in range(population.size):
            members = np.append(population.members, self._archive)

            r1, r2 = rank_selection(members, self._k, self.optimization_type)

            p = 0.085 + (0.085 * nfe) / nfe_max
            how_many_best_to_select = p * pop_size
            how_many_best_to_select = max(1, how_many_best_to_select)  # at least one best member also for pbest it
            # is require minimum 4 members for selection: current, best, r1, r2

            best_members = population.get_best_members(int(how_many_best_to_select))
            all_best_members = self._get_best_members(best_members,
                                                      self._archive)  # get best members from the population and archive

            rand_index = randint(0, len(all_best_members) - 1)

            # First member in argument is the actual member, second is from the best member from the population and archive,
            # third and fourth are random members from the population and archive using the rank selection
            new_member = MutationCurrentToPBestR.mutate(population.members[i], all_best_members[rand_index],
                                                        members[r1],
                                                        members[r2], f_table[i], fw_table[i])

            new_members.append(new_member)

        new_population = Population(
            lb=population.lb,
            ub=population.ub,
            arg_num=population.arg_num,
            size=population.size,
            optimization=population.optimization
        )
        new_population.members = np.array(new_members)
        return new_population

    def _selection(self, origin_population: Population, modified_population: Population, cr_table: List[float],
                   f_table: List[float]):
        """
        Perform selection operation for the population.

        Parameters:
        - origin_population (Population): The original population.
        - modified_population (Population): The modified population
        - cr_table (List[float]): A list of crossover rates Cr for the current epoch.
        - f_table (List[float]): A list of scaling factors F for the current epoch.

        Returns: A new population with the selected members.
        """

        optimization = origin_population.optimization
        new_members = []
        for i in range(origin_population.size):
            if optimization == OptimizationType.MINIMIZATION:
                if origin_population.members[i] < modified_population.members[i]:
                    new_members.append(copy.deepcopy(origin_population.members[i]))
                else:
                    self._archive.append(copy.deepcopy(origin_population.members[i]))
                    new_members.append(copy.deepcopy(modified_population.members[i]))
                    self._success_f.append(f_table[i])
                    self._success_cr.append(cr_table[i])

                    self._difference_fitness_success.append(
                        origin_population.members[i].fitness_value - modified_population.members[i].fitness_value)
            elif optimization == OptimizationType.MAXIMIZATION:
                if origin_population.members[i] > modified_population.members[i]:
                    new_members.append(copy.deepcopy(origin_population.members[i]))
                else:
                    self._archive.append(copy.deepcopy(origin_population.members[i]))
                    new_members.append(copy.deepcopy(modified_population.members[i]))
                    self._success_f.append(f_table[i])
                    self._success_cr.append(cr_table[i])
                    self._difference_fitness_success.append(
                        modified_population.members[i].fitness_value - origin_population.members[i].fitness_value)

        new_population = Population(
            lb=origin_population.lb,
            ub=origin_population.ub,
            arg_num=origin_population.arg_num,
            size=origin_population.size,
            optimization=origin_population.optimization
        )
        new_population.members = np.array(new_members)
        return new_population

    def _update_population_size(self, nfe: int, total_nfe: int, start_pop_size: int, min_pop_size: int):
        """
        Calculate new population size using Linear Population Size Reduction (LPSR).

        Parameters:
        - start_pop_size (int): The initial population size.
        - nfe (int): The current nfe.
        - total_nfe (int): The total number of nfe.
        - min_pop_size (int): The minimum population size.
        """
        new_size = self._population_size_reduction_strategy.get_new_population_size(
            nfe, total_nfe, start_pop_size, min_pop_size
        )

        self._pop.resize(new_size)

        # Update archive size proportionally
        self._archive_size = new_size

    def _calculate_factors_for_epoch(self, pop_size):
        """
        Calculate the mutation parameters (F, Cr, Fw) for the current epoch based on the LSHADE-RSP algorithm.

        Parameters:
        - pop_size (int): The population size.
        """
        f_table = []
        cr_table = []
        fw_table = []

        for i in range(pop_size):
            ri = np.random.randint(0, self._H)
            mean_f = self._memory_F[ri]
            mean_cr = self._memory_Cr[ri]

            if np.isnan(self._memory_Cr[ri]):
                cr = 0.0
            else:
                # Generate Cr from normal distribution
                cr = np.random.normal(mean_cr, 0.1)
                cr = np.clip(cr, 0.0, 1.0)  # Clip to [0, 1]

            # Generate F from Cauchy distribution (ensure F > 0)
            while True:
                f = np.random.standard_cauchy() * 0.1 + mean_f
                if f > 0:
                    break

            # Constrain F based on NFE
            if self.nfe < 0.6 * self.nfe_max:
                f = min(f, 0.7)  # F <= 0.7 for the first 60% of evaluations
            else:
                f = min(f, 1.0)  # F <= 1.0 for the remaining evaluations

            f_table.append(f)
            cr_table.append(cr)

            # Set Fw based on the current NFE
            if self.nfe < 0.2 * self.nfe_max:
                fw = 0.7 * f
            elif self.nfe < 0.4 * self.nfe_max:
                fw = 0.8 * f
            else:
                fw = 1.2 * f
            fw_table.append(fw)

        return f_table, cr_table, fw_table

    def next_epoch(self):
        """
        Perform the next epoch of the L-SHADE-RSP algorithm.
        """

        f_table, cr_table, fw_table = self._calculate_factors_for_epoch(self._pop.size)

        mutant = self._mutate(self._pop, self.nfe, self.nfe_max, self._pop.size, f_table, fw_table)

        trial = crossing(self._pop, mutant, cr_table)

        fix_boundary_constraints_with_parent(self._pop, trial, self.boundary_constraints_fun)

        trial.update_fitness_values(self._function.eval, self.parallel_processing)

        new_pop = self._selection(self._pop, trial, cr_table, f_table)

        archive_reduction(self._archive, self._archive_size, self.optimization_type)

        # Override the entire population with the new population
        self._pop = new_pop

        self._adapt_parameters(self._difference_fitness_success)
        self._update_population_size(self.nfe, self.nfe_max, self._start_population_size,
                                     self._min_pop_size)
