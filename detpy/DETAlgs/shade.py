import copy
from typing import List

import numpy as np
from detpy.DETAlgs.base import BaseAlg
from detpy.DETAlgs.data.alg_data import  ShadeData

from random import randrange

from detpy.models.enums.boundary_constrain import fix_boundary_constraints
from detpy.models.enums.optimization import OptimizationType
from detpy.models.member import Member
from detpy.models.population import Population


def archive_reduction(archive: list[Member], archive_size: int, pop_size: int):
    """
    Reduce the size of the archive to the specified size.

    Parameters:
    - archive (list[Member]): The archive of members from previous populations.
    - archive_size (int): The desired size of the archive.
    - pop_size (int): The size of the population.

    Returns: The reduced archive.
    """
    if archive_size == 0:
        archive.clear()

    max_elem = min(archive_size, pop_size)

    reduce_num = len(archive) - max_elem
    for _ in range(reduce_num):
        idx = randrange(len(archive))
        archive.pop(idx)

    return archive


def mutation_internal(base_member: Member, best_member: Member, r1: Member, r2: Member, f: float):
    """
    Formula: bm + Fw * (bm_best - bm) + F * (r1 - r2)

    Parameters:
    - base_member (Member): The base member used for the mutation operation.
    - best_member (Member): The best member, typically the one with the best fitness value, used in the mutation formula.
    - r1 (Member): A randomly selected member from the population, used for mutation. (rank selection)
    - r2 (Member): Another randomly selected member from archive, used for mutation
    - f (float): A scaling factor that controls the magnitude of the mutation between random members of the population.

    Returns: A new member with the mutated chromosomes.
    """
    new_member = copy.deepcopy(base_member)
    new_member.chromosomes = base_member.chromosomes + (
            f * (best_member.chromosomes - base_member.chromosomes)) + (
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
    - cr_table (List[float]): List of crossover rates.

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


class SHADE(BaseAlg):
    """
        SHADE: Success-History based Adaptive Differential Evolution

        References:
        Ryoji Tanabe and Alex Fukunaga Graduate School of Arts and Sciences The University of Tokyo
    """

    def __init__(self, params: ShadeData, db_conn=None, db_auto_write=False):
        super().__init__(SHADE.__name__, params, db_conn, db_auto_write)

        self.H = params.parameter_memory_size  # Memory size for f and cr adaptation
        self.memory_F = np.full(self.H, 0.5)  # Initial memory for F
        self.memory_Cr = np.full(self.H, 0.5)  # Initial memory for Cr

        self.successCr = []
        self.successF = []
        self.difference_fitness_success = []

        self.min_the_best_percentage = 2 / self.population_size  # Minimal percentage of the best members to consider

        self.archive_size = params.archive_size  # Size of the archive
        self.archive = []  # Archive for storing the members from old populations

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
               the_best_to_select_table: List[int],
               f_table: List[float]
               ) -> Population:
        """
        Perform mutation step for the population in SHADE.

        Parameters:
        - population (Population): The population to mutate.
        - the_best_to_select_table (List[int]): List of the number of the best members to select.
        - f_table (List[float]): List of scaling factors for mutation.

        Returns: A new Population with mutated members.
        """
        new_members = []

        sum_archive_and_population = np.concatenate((population.members, self.archive))

        for i in range(population.size):
            r1 = np.random.choice(len(population.members), 1, replace=False)[0]

            # Archive is included population and archive members
            r2 = np.random.choice(len(sum_archive_and_population), 1, replace=False)[0]

            # Select top p-best members from the population
            best_members = population.get_best_members(the_best_to_select_table[i])

            # Randomly select one of the p-best members
            selected_best_member = best_members[np.random.randint(0, len(best_members))]

            # Apply the mutation formula (current-to-pbest strategy)
            mutated_member = mutation_internal(
                base_member=population.members[i],
                best_member=selected_best_member,
                r1=population.members[r1],
                r2=sum_archive_and_population[r2],
                f=f_table[i]
            )

            new_members.append(mutated_member)

        # Create a new population with the mutated members
        new_population = Population(
            lb=population.lb,
            ub=population.ub,
            arg_num=population.arg_num,
            size=population.size,
            optimization=population.optimization
        )
        new_population.members = np.array(new_members)
        return new_population

    def selection(self, origin_population: Population, modified_population: Population, ftable: List[float],
                  cr_table: List[float]):
        """
        Perform selection operation for the population.

        Parameters:
        - origin_population (Population): The original population.
        - modified_population (Population): The modified population
        - ftable (List[float]): List of scaling factors for mutation.
        - cr_table (List[float]): List of crossover rates.

        Returns: A new population with the selected members.
        """
        optimization = origin_population.optimization
        new_members = []
        for i in range(origin_population.size):
            if optimization == OptimizationType.MINIMIZATION:
                if origin_population.members[i] <= modified_population.members[i]:
                    new_members.append(copy.deepcopy(origin_population.members[i]))
                else:
                    self.archive.append(copy.deepcopy(origin_population.members[i]))
                    self.successF.append(ftable[i])
                    self.successCr.append(cr_table[i])
                    self.difference_fitness_success.append(
                        origin_population.members[i].fitness_value - modified_population.members[i].fitness_value)
                    new_members.append(copy.deepcopy(modified_population.members[i]))
            elif optimization == OptimizationType.MAXIMIZATION:
                if origin_population.members[i] >= modified_population.members[i]:
                    new_members.append(copy.deepcopy(origin_population.members[i]))
                else:
                    self.archive.append(copy.deepcopy(origin_population.members[i]))
                    self.successF.append(ftable[i])
                    self.successCr.append(cr_table[i])
                    self.difference_fitness_success.append(
                        modified_population.members[i].fitness_value - origin_population.members[i].fitness_value)
                    new_members.append(copy.deepcopy(modified_population.members[i]))

        new_population = Population(
            lb=origin_population.lb,
            ub=origin_population.ub,
            arg_num=origin_population.arg_num,
            size=origin_population.size,
            optimization=origin_population.optimization
        )
        new_population.members = np.array(new_members)
        return new_population

    def update_memory(self, success_f, success_cf, difference_fitness_success):
        """
        Update the memory for the crossover rates and scaling factors based on the success of the trial vectors.

        Parameters:
        - success_f (List[float]): List of scaling factors that led to better trial vectors.
        - success_cf (List[float]): List of crossover rates that led to better trial vectors.
        - difference_fitness_success (List[float]): List of differences in objective function values (|f(u_k, G) - f(x_k, G)|).
        """
        if len(success_f) > 0:
            total = np.sum(difference_fitness_success)
            weights = difference_fitness_success / total
            f_new = np.sum(weights * success_f * success_f) / np.sum(weights * success_f)
            f_new = np.clip(f_new, 0, 1)
            cr_new = np.sum(weights * success_cf)
            cr_new = np.clip(cr_new, 0, 1)

            self.memory_F = np.append(self.memory_F[1:], f_new)
            self.memory_Cr = np.append(self.memory_Cr[1:], cr_new)

            # Reset the lists for the next generation
            self.successF = []
            self.successCr = []
            self.difference_fitness_success = []

    @staticmethod
    def calculate_best_member_count(population_size):
        """
        Calculate the number of the best members to select based on a percentage of the population size.
        The percentage is randomly chosen between a minimum value (2/population_size) and a maximum value (20% of the population).

        Parameters:
        - population_size (int): The size of the population.

        Returns:
        - int: The number of the best members to select.
        """

        min_percentage = 2 / population_size
        max_percentage = 0.2

        random_percentage = np.random.uniform(min_percentage, max_percentage)

        return int(random_percentage * population_size)

    def initialize_parameters_for_epoch(self):
        """
        Initialize the parameters for the next epoch of the SHADE algorithm.
        f_table: List of scaling factors for mutation.
        cr_table: List of crossover rates.
        the_bests_to_select: List of the number of the best members to select because in crossover we need to select
        the best members from the population for one factor.

        Returns:
        - f_table (List[float]): List of scaling factors for mutation.
        - cr_table (List[float]): List of crossover rates.
        - the_bests_to_select (List[int]): List of the number of the best members to possibly select.
        """
        f_table = []
        cr_table = []
        the_bests_to_select = []

        for i in range(self._pop.size):
            ri = np.random.randint(0, self.H)
            mean_f = self.memory_F[ri]
            mean_cr = self.memory_Cr[ri]

            f = np.random.normal(mean_f, 0.1)
            cr = np.random.standard_cauchy() * 0.1 + mean_cr

            f = np.clip(f, 0.0, 1.0)
            cr = np.clip(cr, 0.0, 1.0)
            f_table.append(f)
            cr_table.append(cr)

            the_best_to_possible_select = SHADE.calculate_best_member_count(
                self.population_size)

            the_bests_to_select.append(the_best_to_possible_select)

        return f_table, cr_table, the_bests_to_select

    def next_epoch(self):
        """
        Perform the next epoch of the SHADE algorithm.
        """
        f_table, cr_table, the_bests_to_select = self.initialize_parameters_for_epoch()

        mutant = self.mutate(self._pop, the_bests_to_select, f_table)

        fix_boundary_constraints(mutant, self.boundary_constraints_fun)

        # Crossover step
        trial = crossing(self._pop, mutant, cr_table)

        # Evaluate fitness values for the trial population
        trial.update_fitness_values(self._function.eval, self.parallel_processing)

        # Selection step
        new_pop = self.selection(self._pop, trial, f_table, cr_table)

        # Archive management
        self.archive = archive_reduction(self.archive, self.archive_size, self.population_size)

        # Update the population
        self._pop = new_pop

        # Update the memory for CR and F
        self.update_memory(self.successF, self.successCr, self.difference_fitness_success)

        self._epoch_number += 1
