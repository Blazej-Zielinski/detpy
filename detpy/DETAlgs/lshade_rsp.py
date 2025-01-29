import copy
from typing import List

import numpy as np
from detpy.DETAlgs.base import BaseAlg
from detpy.DETAlgs.data.alg_data import LShadeRSPData

from random import randint, randrange

from detpy.DETAlgs.math_functions.lehmer_mean import LehmerMean
from detpy.models.enums.boundary_constrain import fix_boundary_constraints
from detpy.models.enums.optimization import OptimizationType
from detpy.models.member import Member
from detpy.models.population import Population


def archive_reduction(archive: list[Member], archive_size: int):
    """
    Reduce the size of the archive to the specified size by removing the worst members.
    """
    if archive_size == 0:
        archive.clear()
        return

    if len(archive) > archive_size:
        # Sort archive by fitness (worst first)
        archive.sort(key=lambda member: member.fitness_value, reverse=True)
        # Remove the worst members
        del archive[archive_size:]


def mutation_internal(base_member: Member, best_member: Member, r1: Member, r2: Member, f: float, fw: float):
    """
    Formula: bm + Fw * (bm_best - bm) + F * (r1 - r2)

    Parameters:
    - base_member (Member): The base member used for the mutation operation.
    - best_member (Member): The best member, typically the one with the best fitness value, used in the mutation formula.
    - r1 (Member): A randomly selected member from the population, used for mutation. (rank selection)
    - r2 (Member): Another randomly selected member, used for mutation. (rank selection)
    - f (float): A scaling factor that controls the magnitude of the mutation between random members of the population.
    - fw (float): A scaling factor that controls the magnitude of the mutation between the best member and the base member.

    Returns: A new member with the mutated chromosomes.
    """
    new_member = copy.deepcopy(base_member)
    new_member.chromosomes = base_member.chromosomes + (
            fw * (best_member.chromosomes - base_member.chromosomes)) + (
                                     f * (r1.chromosomes - r2.chromosomes))
    return new_member


def crossing_internal(org_member: Member, mut_member: Member, cr: float):
    """
    Perform crossing operation for two members.

    Parameters:
    - org_member (Member): The original member.
    - mut_member (Member): The mutated member.
    - cr (float): The crossover rate.

    Returns: A new member with the crossed chromosomes.
    """
    new_member = copy.deepcopy(org_member)

    random_numbers = np.random.rand(new_member.args_num)
    mask = random_numbers <= cr

    # ensures that new member gets at least one parameter
    i_rand = np.random.randint(low=0, high=new_member.args_num)

    for i in range(new_member.args_num):
        if mask[i] or i_rand == i:
            new_member.chromosomes[i].real_value = mut_member.chromosomes[i].real_value
        else:
            new_member.chromosomes[i].real_value = org_member.chromosomes[i].real_value

    return new_member


def crossing(origin_population: Population, mutated_population: Population, cr_table):
    """
    Perform crossing operation for the population.
    Parameters:
    - origin_population (Population): The original population.
    - mutated_population (Population): The mutated population.
    - cr (float): The crossover rate.

    Returns: A new population with the crossed chromosomes.
    """
    if origin_population.size != mutated_population.size:
        print("Binomial_crossing: populations have different sizes")
        return None

    new_members = []
    for i in range(origin_population.size):
        new_member = crossing_internal(origin_population.members[i], mutated_population.members[i], cr_table[i])
        new_members.append(new_member)

    new_population = Population(
        lb=origin_population.lb,
        ub=origin_population.ub,
        arg_num=origin_population.arg_num,
        size=origin_population.size,
        optimization=origin_population.optimization
    )
    new_population.members = np.array(new_members)
    return new_population


def rank_selection(members: List[Member], k: float = 1.0) -> (int, int):
    """
    Choose two random members from the population using rank selection.
    Parameters:
    - members (Population): The population from which to select the members.
    - k (float): Scaling factor for rank selection.

    Returns: The indexes of the selected members.
    """
    sorted_population_member_indexes = np.argsort([member.fitness_value for member in members])
    values = list(range(len(members), 0, -1))
    ranks = k * np.array(values)
    probabilities = ranks / ranks.sum()

    # Ensure distinct indices
    selected_indices = np.random.choice(len(sorted_population_member_indexes), size=2, replace=False, p=probabilities)
    selected_values = sorted_population_member_indexes[selected_indices]

    return int(selected_values[0]), int(selected_values[1])


class LSHADERSP(BaseAlg):
    """
        L-SHADE-RSP: LSHADE Algorithm with a Rank-based Selective Pressure Strategy (RSP)

        References:
        Shakhnaz Akhmedova, Vladimir Stanovov and Eugene Semenkin (2018). Research on In Proceedings of
        the 15th International Conference on Informatics in Control, Automation and Robotics (ICINCO 2018) -
        Volume 1, pages 149-155
    """

    def __init__(self, params: LShadeRSPData, db_conn=None, db_auto_write=False):
        super().__init__(LSHADERSP.__name__, params, db_conn, db_auto_write)
        self.k = params.scaling_factor_for_rank_selection  # Scaling factor for rank selection
        self.H = params.parameter_memory_size  # Memory size for f and cr adaptation
        self.memory_F = np.full(self.H, 0.3)  # Initial memory for F
        self.memory_Cr = np.full(self.H, 0.8)  # Initial memory for Cr

        self.memory_F[self.H - 1] = 0.9  # One cell of the memory for F must be set to 0.9
        self.memory_Cr[self.H - 1] = 0.9  # One cell of the memory for Cr must be set to 0.9

        self.F = np.random.standard_cauchy() * 0.1 + np.random.choice(self.memory_F)  # The scaling factor for

        self.nfe = 0  # Number of function evaluations
        self.min_pop_size = params.minimum_population_size  # Minimal population size

        self.start_population_size = self.population_size
        self.nfe_max = self.calculate_max_evaluations_lpsr(self.population_size)  # Max number of function evaluations

        self.archive_size = params.archive_size  # Size of the archive
        self.archive = []  # Archive for storing the members from old populations

        self.lehmer_mean_func = LehmerMean()  # Class for Lehmer mean calculation

        self.success_f = []  # List of successful F values
        self.success_cr = []  # List of successful Cr values

        self.F = np.random.standard_cauchy() * 0.1 + np.random.choice(self.memory_F)
        self.Fw = 0.7 * self.F if self.nfe < 0.2 * self.nfe_max else (
            0.8 * self.F if self.nfe < 0.4 * self.nfe_max else 1.2 * self.F)

        self.difference_fitness_success = []

    def calculate_max_evaluations_lpsr(self, start_pop_size):
        """
        Calculate the maximum number of function evaluations for the Linear Population Size Reduction (LPSR) method.

        Parameters:
        - start_pop_size (int): The initial population size at the beginning of the evolution process.

        Returns: sum of the total number of evaluations for each generation.
        """
        total_evaluations = 0
        for generation in range(self.num_of_epochs):
            current_population_size = int(
                start_pop_size - (generation / self.num_of_epochs) * (start_pop_size - self.min_pop_size)
            )
            total_evaluations += current_population_size
        NFEmax = total_evaluations
        return NFEmax

    def adapt_parameters(self, fitness_improvement: List[float]):
        """
        Update the memory for F and Cr based on the fitness improvement of the crossover and mutation steps.

        Parameters:
        - fitness_improvement: List of fitness improvement values for successful individuals.
        """
        epsilon = 1e-10  # Small value to avoid division by zero
        if sum(fitness_improvement) > epsilon:
            # Compute weights based on fitness improvement
            total_improvement = np.sum(fitness_improvement)
            weights = fitness_improvement / total_improvement

            # Compute new F value as the Lehmer mean of successful F values
            if len(self.success_f) > 0:
                new_f = self.lehmer_mean_func.evaluate(self.success_f, weights)
                new_f = np.clip(new_f, 0, 1)  # Clip to [0, 1]
            else:
                new_f = np.random.uniform(0, 1)  # Fallback if no successful F values

            # Compute new Cr value as the weighted arithmetic mean of successful Cr values
            if len(self.success_cr) > 0:
                new_cr = np.sum(weights * self.success_cr)
                new_cr = np.clip(new_cr, 0, 1)  # Clip to [0, 1]
            else:
                new_cr = np.random.uniform(0, 1)  # Fallback if no successful Cr values

            # Randomly select a memory index to update
            r = np.random.randint(0, self.H)

            # Update memory_F and memory_Cr
            self.memory_F[r] = (self.memory_F[r] + new_f) / 2  # Mean of old and new F
            self.memory_Cr[r] = (self.memory_Cr[r] + new_cr) / 2  # Mean of old and new Cr

            # Ensure one cell in the memory always holds the fixed value 0.9
            self.memory_F[0] = 0.9
            self.memory_Cr[0] = 0.9

        # Clear the lists of successful F and Cr values for the next generation
        self.difference_fitness_success = []
        self.success_cr = []
        self.success_f = []

    def get_best_members(self, best_members: List[Member], archive: List[Member]) -> List[Member]:
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

    def mutate(self,
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
            members = np.append(population.members, self.archive)

            r1, r2 = rank_selection(members, self.k)
            p = 0.085 + (0.085 * nfe) / nfe_max
            how_many_best_to_select = p * pop_size
            how_many_best_to_select = max(1, how_many_best_to_select)  # at least one best member also for pbest it
            # is require minimum 4 members for selection: current, best, r1, r2

            best_members = population.get_best_members(int(how_many_best_to_select))
            all_best_members = self.get_best_members(best_members,
                                                     self.archive)  # get best members from the population and archive

            rand_index = randint(0, len(all_best_members) - 1)

            # First member in argument is the actual member, second is from the best member from the population and archive,
            # third and fourth are random members from the population and archive using the rank selection
            new_member = mutation_internal(population.members[i], all_best_members[rand_index], members[r1],
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

    def selection(self, origin_population: Population, modified_population: Population, cr_table, ftable):
        """
        Perform selection operation for the population.

        Parameters:
        - origin_population (Population): The original population.
        - modified_population (Population): The modified population

        Returns: A new population with the selected members.
        """

        optimization = origin_population.optimization
        new_members = []
        for i in range(origin_population.size):
            if optimization == OptimizationType.MINIMIZATION:
                if origin_population.members[i] < modified_population.members[i]:
                    new_members.append(copy.deepcopy(origin_population.members[i]))
                else:
                    self.archive.append(copy.deepcopy(origin_population.members[i]))
                    new_members.append(copy.deepcopy(modified_population.members[i]))
                    self.success_f.append(ftable[i])
                    self.success_cr.append(cr_table[i])

                    self.difference_fitness_success.append(
                        origin_population.members[i].fitness_value - modified_population.members[i].fitness_value)
            elif optimization == OptimizationType.MAXIMIZATION:
                if origin_population.members[i] > modified_population.members[i]:
                    new_members.append(copy.deepcopy(origin_population.members[i]))
                else:
                    self.archive.append(copy.deepcopy(origin_population.members[i]))
                    new_members.append(copy.deepcopy(modified_population.members[i]))
                    self.success_f.append(ftable[i])
                    self.success_cr.append(cr_table[i])
                    self.difference_fitness_success.append(
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

    def update_population_size(self, start_pop_size: int, epoch: int, total_epochs: int, min_size: int):
        """
        Calculate new population size using Linear Population Size Reduction (LPSR).
        """
        new_size = int(start_pop_size - (epoch / total_epochs) * (start_pop_size - min_size))
        self._pop.resize(new_size)

        # Update archive size proportionally
        self.archive_size = int(self.archive_size * (new_size / start_pop_size))

    def calculate_factors_for_epoch(self, pop_size):
        """
        Calculate the mutation parameters (F, Cr, Fw) for the current epoch based on the LSHADE-RSP algorithm.
        """
        f_table = []
        cr_table = []
        fw_table = []

        for i in range(pop_size):
            ri = np.random.randint(0, self.H)
            mean_f = self.memory_F[ri]
            mean_cr = self.memory_Cr[ri]

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

        f_table, cr_table, fw_table = self.calculate_factors_for_epoch(self._pop.size)

        mutant = self.mutate(self._pop, self.nfe, self.nfe_max, self._pop.size, f_table, fw_table)

        fix_boundary_constraints(mutant, self.boundary_constraints_fun)

        trial = crossing(self._pop, mutant, cr_table)

        trial.update_fitness_values(self._function.eval, self.parallel_processing)

        new_pop = self.selection(self._pop, trial, cr_table, f_table)

        archive_reduction(self.archive, self.archive_size)
        print("Archive size: ", len(self.archive))

        # Override the entire population with the new population
        self._pop = new_pop

        self.adapt_parameters(self.difference_fitness_success)
        self.update_population_size(self.start_population_size, self._epoch_number, self.num_of_epochs,
                                    self.min_pop_size)
        self.nfe += self.population_size

        self._epoch_number += 1
