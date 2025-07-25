import copy
import random
from typing import List

import numpy as np
from scipy.optimize import minimize
from scipy.stats import iqr

from detpy.DETAlgs.base import BaseAlg
from detpy.DETAlgs.data.alg_data import DEACRData
from detpy.DETAlgs.methods.methods_de import mutation, mutation_ind, binomial_crossing_ind, selection
from detpy.models.enums.boundary_constrain import fix_boundary_constraints, get_boundary_constraints_fun
from detpy.models.enums.optimization import OptimizationType
from detpy.models.member import Member
from detpy.models.population import Population

class DETCR(BaseAlg):
    """
          DETCR - Hybrid DE Algorithm With Adaptive Crossover Operator

          Links:
          https://ieeexplore.ieee.org/document/5949800

          References:
          Gilberto Reynoso-Meza; Javier Sanchis; Xavier Blasco; Juan M. Herrero,
          "Hybrid DE algorithm with adaptive crossover operator for solving real-world numerical optimization problems",
          2011 IEEE Congress of Evolutionary Computation (CEC),
          New Orleans, LA, USA, 2011, doi: 10.1109/CEC.2011.5949800.
    """

    def __init__(self, params: DEACRData, db_conn=None, db_auto_write=False):
        super().__init__(DETCR.__name__, params, db_conn, db_auto_write)

        self.rate_ls = 1 - (1 / (100 * self.nr_of_args))

        #Adaptive mechanism
        self.max_crossover_rate = params.triangular_distribution_for_crossover_rate[2]
        self.med_crossover_rate = params.triangular_distribution_for_crossover_rate[1]
        self.min_crossover_rate = params.triangular_distribution_for_crossover_rate[0]

        self.max_mutation_factory = params.triangular_distribution_for_mutation_factory[2]
        self.med_mutation_factory = params.triangular_distribution_for_mutation_factory[1]
        self.min_mutation_factory = params.triangular_distribution_for_mutation_factory[0]

        #Population management
        self.gamma_var = params.gamma_var
        self.min_diversity_in_population = 0.05 * (np.array(self._pop.ub) - np.array(self._pop.lb))
        self.population_refreshment_size = 2 * self.nr_of_args

        self.success_evo_cr = 0
        self.number_of_success_crossover_rate = params.number_of_success_crossover_rate
        self.lineal_recombination_factor = params.lineal_recombination_factor
        self.optimization_bounds = np.column_stack((self.lb, self.ub))

    def mutation_ind(self, base_member: Member, member1: Member, member2: Member, f):
        new_member = copy.deepcopy(base_member)
        new_member.chromosomes = base_member.chromosomes + (member1.chromosomes - member2.chromosomes) * f
        return new_member

    def mutation(self, population: Population, f: List):
        new_members = []
        for i in range(population.size):
            selected_members = random.sample(population.members.tolist(), 3)
            new_member = self.mutation_ind(selected_members[0], selected_members[1], selected_members[2], f[i])
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

    def local_search(self, member: Member, function_eval, optimization: OptimizationType):
        local_search_member = copy.deepcopy(member)
        bounds = list(zip(local_search_member.lb, local_search_member.ub))

        options = {
            'maxfun': int(1e4),
            'disp': False
        }

        res = None
        if optimization == OptimizationType.MINIMIZATION:
            res = minimize(
                lambda chromosomes: function_eval(chromosomes),
                member.get_chromosomes(),
                method='SLSQP',
                bounds=bounds,
                options=options)
        else:
            res = minimize(
                lambda chromosomes: -function_eval(chromosomes),
                member.get_chromosomes(),
                method='SLSQP',
                bounds=bounds,
                options=options)

        if not np.any(np.isnan(res.x)):
            local_search_member.chromosomes = res.x

        return local_search_member, res.nfev

    def lineal_recombination(self, org_member: Member, mut_member: Member, linear_recombination):
        new_member = copy.deepcopy(org_member)

        for i in range(new_member.args_num):
            new_member.chromosomes[i] = org_member.chromosomes[i] + linear_recombination * (
                        mut_member.chromosomes[i] - org_member.chromosomes[i])

        return new_member

    def binomial_recombination(self, org_member: Member, mut_member: Member, cr):
        new_member = copy.deepcopy(org_member)
        mutation_success_count = 0
        no_change_counter = 0
        for i in range(new_member.args_num):
            if random.random() > cr:
                new_member.chromosomes[i] = org_member.chromosomes[i]
                mutation_success_count = 1
                no_change_counter += 1
            else:
                new_member.chromosomes[i] = mut_member.chromosomes[i]

        return new_member, mutation_success_count, no_change_counter

    def crossing(self, origin_population: Population, mutated_population: Population, cr_list: List, linear_recombination):
        if origin_population.size != mutated_population.size:
            print("Binomial_crossing: populations have different sizes")
            return None

        new_members = []
        boundary_constraints_fun = get_boundary_constraints_fun(self.boundary_constraints_fun)
        optimization = origin_population.optimization
        for i in range(origin_population.size):
            mutation_success_count = 0
            no_change_counter = 0
            cr = cr_list[0]
            new_member = None

            if cr >= 0.95:
                new_member = self.lineal_recombination(origin_population.members[i], mutated_population.members[i],
                                                 linear_recombination)
                mutation_success_count = 1
            else:
                new_member, mutation_success_count, no_change_counter \
                    = self.binomial_recombination(origin_population.members[i], mutated_population.members[i], cr)

            if mutation_success_count == 0 and cr < 0.5:
                permutation_first_value = np.random.permutation(self.nr_of_args)[0]
                new_member.chromosomes[permutation_first_value] = origin_population.members[i].chromosomes[permutation_first_value]

            if mutation_success_count == 0 and 0.5 <= cr < 0.95:
                new_member.chromosomes = (origin_population.members[i].chromosomes + linear_recombination
                              * (mutated_population.members[i].chromosomes - origin_population.members[i].chromosomes))

            if no_change_counter == self.nr_of_args:
                permutation_first_value = np.random.permutation(self.nr_of_args)
                new_member.chromosomes[permutation_first_value] = mutated_population.members[i].chromosomes[permutation_first_value]

            if not new_member.is_member_in_interval():
                boundary_constraints_fun(new_member)

            if random.random() > self.rate_ls:
                new_member, function_evaluation = self.local_search(new_member, self._function.eval, optimization)

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

    def population_refreshment_mechanism(self, pop : Population):
        refresh_population = copy.deepcopy(pop)
        members = np.array([member.get_chromosomes() for member in pop.members])
        var_pop = iqr(members, axis=0)
        if np.all(var_pop <= ((self.optimization_bounds[:, 1] - self.optimization_bounds[:, 0]) / self.gamma_var)):
            maximization = True if self.optimization_type == OptimizationType.MAXIMIZATION else False
            looking_best = sorted(pop.members, key=lambda member: member.fitness_value, reverse=maximization)
            median = np.median(members, axis=0)
            if np.all(var_pop < self.min_diversity_in_population):
                self.optimization_bounds[:, 0] = median - self.min_diversity_in_population
                self.optimization_bounds[:, 1] = median + self.min_diversity_in_population
            else:
                median = np.median(members, axis = 0)
                self.optimization_bounds[:, 0] = median - (self.optimization_bounds[:, 1] - self.optimization_bounds[:, 0]) / self.gamma_var
                self.optimization_bounds[:, 1] = median + (self.optimization_bounds[:, 1] - self.optimization_bounds[:, 0]) / self.gamma_var

            self.check_optimization_bounds()

            refresh_size = self.population_size - self.population_refreshment_size

            for i in range(self.population_size):
                for j in range(self.nr_of_args):
                    if i < refresh_size:
                        refresh_population.members[i].chromosomes[j].real_value \
                            = (self.optimization_bounds[j, 0] +
                               (self.optimization_bounds[j, 1] - self.optimization_bounds[j, 0]) * random.random())
                    else:
                        refresh_population.members[i].chromosomes[j].real_value \
                            = looking_best[i - refresh_size].chromosomes[j].real_value

        return refresh_population

    def check_optimization_bounds(self):
        for i in range(self.nr_of_args):
            if self.optimization_bounds[i, 0] < self.lb[i]:
                self.optimization_bounds[i, 0] = self.lb[i]
            if self.optimization_bounds[i, 1] > self.ub[i]:
                self.optimization_bounds[i, 1] = self.ub[i]

    def adaptive_mechanism(self):
        mutation_factories = np.zeros(self.population_size)
        crossover_rates = np.zeros(self.population_size)
        for i in range(self.population_size):
            rand_u = np.random.rand()
            if rand_u < (self.med_mutation_factory - self.min_mutation_factory) * (
                    2 / (self.max_mutation_factory - self.min_mutation_factory)) * 0.5:
                mutation_factories[i] = self.min_mutation_factory + np.sqrt(
                    rand_u * (self.max_mutation_factory - self.min_mutation_factory) * (
                                self.med_mutation_factory - self.min_mutation_factory))
            else:
                mutation_factories[i] = self.max_mutation_factory - np.sqrt(
                    (1 - rand_u) * (self.max_mutation_factory - self.min_mutation_factory) * (
                                self.max_mutation_factory - self.med_mutation_factory))

            rand_u = np.random.rand()
            if rand_u < (self.med_crossover_rate - self.min_crossover_rate) * (
                    2 / (self.max_crossover_rate - self.min_crossover_rate)) * 0.5:
                crossover_rates[i] = self.min_crossover_rate + np.sqrt(
                    rand_u * (self.max_crossover_rate - self.min_crossover_rate) * (
                            self.med_crossover_rate - self.min_crossover_rate))
            else:
                crossover_rates[i] = self.max_crossover_rate - np.sqrt(
                    (1 - rand_u) * (self.max_crossover_rate - self.min_crossover_rate) * (
                                self.max_crossover_rate - self.med_crossover_rate))
        return mutation_factories.tolist(), crossover_rates.tolist()

    def adapting_triangular_distribution(self, mutation_factories : List):
        if self.success_evo_cr >= self.number_of_success_crossover_rate:
            self.success_evo_cr = 0

            self.max_crossover_rate = np.max(mutation_factories)
            self.min_crossover_rate = np.min(mutation_factories)
            self.med_crossover_rate = np.median(mutation_factories)

            if abs(self.med_crossover_rate - self.max_crossover_rate) < 0.1:
                self.max_crossover_rate = min(self.med_crossover_rate + 0.1, 1.0)

            if abs(self.med_crossover_rate - self.min_crossover_rate) < 0.1:
                self.min_crossover_rate = max(self.med_crossover_rate - 0.1, 1.0)

    def selection(self, origin_population: Population, modified_population: Population):
        if origin_population.size != modified_population.size:
            print("Selection: populations have different sizes")
            return None

        if origin_population.optimization != modified_population.optimization:
            print("Selection: populations have different optimization types")
            return None

        optimization = origin_population.optimization
        new_members = []
        for i in range(origin_population.size):
            if optimization == OptimizationType.MINIMIZATION:
                if origin_population.members[i] <= modified_population.members[i]:
                    new_members.append(copy.deepcopy(origin_population.members[i]))
                else:
                    new_members.append(copy.deepcopy(modified_population.members[i]))
                    self.success_evo_cr += 1
            elif optimization == OptimizationType.MAXIMIZATION:
                if origin_population.members[i] >= modified_population.members[i]:
                    new_members.append(copy.deepcopy(origin_population.members[i]))
                else:
                    new_members.append(copy.deepcopy(modified_population.members[i]))
                    self.success_evo_cr += 1

        new_population = Population(
            lb=origin_population.lb,
            ub=origin_population.ub,
            arg_num=origin_population.arg_num,
            size=origin_population.size,
            optimization=origin_population.optimization
        )
        new_population.members = np.array(new_members)
        return new_population

    def next_epoch(self):
        refresh_population = self.population_refreshment_mechanism(self._pop)

        mutation_factories, crossover_rates = self.adaptive_mechanism()

        v_pop = self.mutation(refresh_population, mutation_factories)

        fix_boundary_constraints(v_pop, self.boundary_constraints_fun)

        # New population after crossing
        u_pop = self.crossing(self._pop, v_pop, crossover_rates, self.lineal_recombination_factor)

        # Update values before selection
        u_pop.update_fitness_values(self._function.eval, self.parallel_processing)

        # Select new population
        new_pop = selection(self._pop, u_pop)

        self.adapting_triangular_distribution(mutation_factories)

        # Override data
        self._pop = new_pop

        self._epoch_number += 1