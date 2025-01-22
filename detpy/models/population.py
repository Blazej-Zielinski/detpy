import concurrent.futures
import numpy as np
from detpy.models.enums.optimization import OptimizationType
from detpy.models.member import Member


class Population:
    def __init__(self, lb, ub, arg_num, size, optimization: OptimizationType):
        self.size = size
        self.members = None
        self.optimization = optimization

        # chromosome config
        self.lb = lb
        self.ub = ub
        self.arg_num = arg_num

    def generate_population(self):
        self.members = np.array([Member(self.lb, self.ub, self.arg_num) for _ in range(self.size)])

    @staticmethod
    def calculate_fitness(member, fitness_fun):
        args = member.get_chromosomes()
        return fitness_fun(args)

    def calculate_member(self, index, fitness_fun):
        member = self.members[index]
        member.fitness_value = self.calculate_fitness(member, fitness_fun)

    def update_fitness_values(self, fitness_fun, parallel_processing=None):
        if parallel_processing is None:
            executor_class = concurrent.futures.ThreadPoolExecutor
            worker = 1
        elif parallel_processing[0] == "process":
            raise ValueError('ProcessPoolExecutor is not supported. Please use thread configuration.')
        else:
            executor_class = concurrent.futures.ThreadPoolExecutor
            worker = parallel_processing[1]

        with executor_class(max_workers=worker) as executor:
            executor.map(lambda idx: self.calculate_member(idx, fitness_fun), range(len(self.members)))

    def get_best_members(self, nr_of_members):
        # Get the indices that would sort the array based on the key function
        sorted_indices = np.argsort([member.fitness_value for member in self.members])
        # Use the sorted indices to sort the array
        sorted_array = self.members[sorted_indices]
        return sorted_array[:nr_of_members]

    def resize(self, new_size: int):
        """
        Resize the population to the new size. If the new size is smaller, truncate the members array by removing the
        worst members. If the new size is larger, add new members with random chromosomes.

        Parameters:
        new_size (int): The new size of the population.
        """
        if new_size < self.size:
            if self.optimization == OptimizationType.MINIMIZATION:
                sorted_indices = np.argsort([member.fitness_value for member in self.members])
            else:
                sorted_indices = np.argsort([member.fitness_value for member in self.members])[::-1]
            self.members = self.members[sorted_indices[:new_size]]
        elif new_size > self.size:
            additional_members = [
                Member(
                    np.random.uniform(self.lb, self.ub, self.arg_num)
                )
                for _ in range(new_size - self.size)
            ]
            self.members = np.concatenate((self.members, additional_members))

        self.size = new_size

    def mean(self):
        return np.mean([member.fitness_value for member in self.members])

    def std(self):
        return np.std([member.fitness_value for member in self.members])

    def __str__(self, population_label=""):
        output = f"Population{population_label}:"
        for m in self.members:
            output += f"\n{str(m)}"

        return output
