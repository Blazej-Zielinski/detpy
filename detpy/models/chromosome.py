import numpy as np
import math
from abc import ABC, abstractmethod


class BaseChromosome(ABC):

    @abstractmethod
    def calculate_real_value(self, bin_ind):
        pass


class Chromosome(BaseChromosome):
    def __init__(self, lb, ub):
        # Options
        self.lb = lb
        self.ub = ub

        # Real value
        self.real_value = np.random.uniform(self.lb, self.ub)

    def calculate_real_value(self, bin_ind):
        binary_string = ''.join([str(elem) for elem in bin_ind])
        return self.lb + int(binary_string, 2) * (self.ub - self.lb) / (
                math.pow(2, bin_ind.size) - 1)

    def __add__(self, other):
        c = Chromosome(self.lb, self.ub)
        c.real_value = self.real_value + other.real_value
        return c

    def __sub__(self, other):
        c = Chromosome(self.lb, self.ub)
        if isinstance(other, Chromosome):
            c.real_value = self.real_value - other.real_value
        elif isinstance(other, (float, np.int32, np.int64, np.float32, np.float64)):
            c.real_value = self.real_value - other
        else:
            raise TypeError(f"Unsupported operand type(s) for -: 'Chromosome' and '{type(other).__name__}'")
        return c

    def __mul__(self, other):
        c = Chromosome(self.lb, self.ub)
        if isinstance(other, Chromosome):
            c.real_value = self.real_value * other.real_value
        elif isinstance(other, (int, float, np.int32, np.int64, np.float32, np.float64)):
            c.real_value = self.real_value * other
        else:
            raise TypeError(f"Unsupported operand type(s) for *: 'Chromosome' and '{type(other).__name__}'")
        return c

    def __rmul__(self, other):
        return self.__mul__(other)

    def __abs__(self):
        c = Chromosome(self.lb, self.ub)
        c.real_value = abs(self.real_value)
        return c
