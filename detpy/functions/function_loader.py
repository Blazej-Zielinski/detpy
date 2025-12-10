import os
import json
import math
import numpy as np
from opfunu.cec_based import F12014, F212014, F12013
from scipy.stats import multivariate_normal

class Ackley:
    def __init__(self, n_dimensions):
        self.n_dimensions = n_dimensions
        self.name="Ackley"

    def evaluate_func(self, x):
        if len(x) != self.n_dimensions:
            raise ValueError(f"Ackley function requires {self.n_dimensions} variables, but {len(x)} were given.")
        a = 20
        b = 0.2
        c = 2 * math.pi
        part1 = -a * math.exp(-b * np.sqrt(np.sum(np.square(x)) / self.n_dimensions))
        part2 = -math.exp(np.sum(np.cos(c * np.array(x))) / self.n_dimensions)
        return part1 + part2 + a + math.exp(1)

class Rastrigin:
    def __init__(self, n_dimensions):
        self.n_dimensions = n_dimensions
        self.name = "Rastrigin"

    def evaluate_func(self, x):
        if len(x) != self.n_dimensions:
            raise ValueError(f"Rastrigin function requires {self.n_dimensions} variables, but {len(x)} were given.")
        A = 10
        return A * self.n_dimensions + sum([(xi ** 2 - A * np.cos(2 * np.pi * xi)) for xi in x])

class Rosenbrock:
    def __init__(self, n_dimensions):
        self.n_dimensions = n_dimensions
        self.name = "Rosenbrock"

    def evaluate_func(self, x):
        if len(x) != self.n_dimensions:
            raise ValueError(f"Rosenbrock function requires {self.n_dimensions} variables, but {len(x)} were given.")
        return sum([100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(self.n_dimensions - 1)])

class Sphere:
    def __init__(self, n_dimensions):
        self.n_dimensions = n_dimensions
        self.name = "Sphere"

    def evaluate_func(self, x):
        if len(x) != self.n_dimensions:
            raise ValueError(f"Sphere function requires {self.n_dimensions} variables, but {len(x)} were given.")
        return sum([xi ** 2 for xi in x])

class Griewank:
    def __init__(self, n_dimensions):
        self.n_dimensions = n_dimensions
        self.name = "Griewank"

    def evaluate_func(self, x):
        if len(x) != self.n_dimensions:
            raise ValueError(f"Griewank function requires {self.n_dimensions} variables, but {len(x)} were given.")
        sum_part = sum([xi ** 2 / 4000 for xi in x])
        prod_part = np.prod([np.cos(xi / np.sqrt(i + 1)) for i, xi in enumerate(x)])
        return sum_part - prod_part + 1

class Schwefel:
    def __init__(self, n_dimensions):
        self.n_dimensions = n_dimensions
        self.name = "Schwefel"

    def evaluate_func(self, x):
        if len(x) != self.n_dimensions:
            raise ValueError(f"Schwefel function requires {self.n_dimensions} variables, but {len(x)} were given.")
        return 418.9829 * self.n_dimensions - sum([xi * np.sin(np.sqrt(abs(xi))) for xi in x])

class Michalewicz:
    def __init__(self, n_dimensions, m=10):
        self.n_dimensions = n_dimensions
        self.m = m
        self.name = "Michalewicz"

    def evaluate_func(self, x):
        if len(x) != self.n_dimensions:
            raise ValueError(f"Michalewicz function requires {self.n_dimensions} variables, but {len(x)} were given.")
        return -sum([math.sin(xi) * (math.sin((i + 1) * xi ** 2 / math.pi) ** (2 * self.m)) for i, xi in enumerate(x)])

class Easom:
    def __init__(self):
        self.n_dimensions = 2
        self.name = "Easom"

    def evaluate_func(self, x):
        if len(x) != self.n_dimensions:
            raise ValueError(f"Easom function requires 2 variables, but {len(x)} were given.")
        return -math.cos(x[0]) * math.cos(x[1]) * math.exp(-((x[0] - math.pi) ** 2 + (x[1] - math.pi) ** 2))

class Himmelblau:
    def __init__(self):
        self.n_dimensions = 2
        self.name = "Himmelblau"

    def evaluate_func(self, x):
        if len(x) != self.n_dimensions:
            raise ValueError(f"Himmelblau function requires 2 variables, but {len(x)} were given.")
        return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2

class Keane:
    def __init__(self, n_dimensions):
        self.n_dimensions = n_dimensions
        self.name = "Keane"

    def evaluate_func(self, x):
        if len(x) != self.n_dimensions:
            raise ValueError(f"Keane function requires {self.n_dimensions} variables, but {len(x)} were given.")
        part0 = np.prod([np.cos(xi) ** 2 for xi in x])
        part1 = abs(sum([np.cos(xi) ** 4 for xi in x]) - 2 * part0)
        part2 = math.sqrt(sum([(i + 1) * xi ** 2 for i, xi in enumerate(x)]))
        return -part1 / part2

class Rana:
    def __init__(self, n_dimensions):
        self.n_dimensions = n_dimensions
        self.name = "Rana"

    def evaluate_func(self, x):
        if len(x) != self.n_dimensions:
            raise ValueError(f"Rana function requires {self.n_dimensions} variables, but {len(x)} were given.")
        s = 0.0
        for i in range(self.n_dimensions - 1):
            s += x[i] * math.cos(math.sqrt(abs(x[i + 1] + x[i] + 1))) * math.sin(math.sqrt(abs(x[i + 1] - x[i] + 1)))
        return s

class PitsAndHoles:
    def __init__(self):
        self.n_dimensions = 2
        self.mu = [[0, 0], [20, 0], [0, 20], [-20, 0], [0, -20], [10, 10], [-10, -10], [-10, 10], [10, -10]]
        self.c = [10.5, 14.0, 16.0, 12.0, 9.0, 0.1, 0.2, 0.25, 0.17]
        self.v = [2.0, 2.5, 2.7, 2.5, 2.3, 0.05, 0.3, 0.24, 0.23]
        self.name = "PitsAndHoles"

    def _get_covariance_matrix(self, idx):
        return [[self.c[idx], 0], [0, self.c[idx]]]

    def evaluate_func(self, x):
        if len(x) != self.n_dimensions:
            raise ValueError(f"Pits and Holes function requires 2 variables, but {len(x)} were given.")
        v = 0
        for i in range(len(self.mu)):
            v += multivariate_normal.pdf(x, mean=self.mu[i], cov=self._get_covariance_matrix(i)) * self.v[i]
        return -v

class CEC2014F1_Opfunu:
    def __init__(self, n_dimensions):
        self.n_dimensions = n_dimensions
        self.name = "CEC2014F1"
        self.function = F12014(ndim=n_dimensions)

    def evaluate_func(self, x):
        if len(x) != self.n_dimensions:
            raise ValueError(f"CEC2014 F1 function requires {self.n_dimensions} variables, but {len(x)} were given.")
        return self.function.evaluate(x)


class CEC2013F1_Opfunu:
    def __init__(self, n_dimensions):
        self.n_dimensions = n_dimensions
        self.name = "CEC2013F1"
        self.function = F12013(ndim=n_dimensions)

    def evaluate_func(self, x):
        if len(x) != self.n_dimensions:
            raise ValueError(f"CEC2013 F1 function requires {self.n_dimensions} variables, but {len(x)} were given.")
        return self.function.evaluate(x)

class CEC2013F1_OpfunuMAX:
    def __init__(self, n_dimensions):
        self.n_dimensions = n_dimensions
        self.name = "CEC2013F1_Maximization"
        self.function = F12013(ndim=n_dimensions)
        self.f_bias = self.function.f_bias        # -1400
        self.f_max_target = -100                  # docelowe globalne maksimum

    def evaluate_func(self, x):
        val_min = self.function.evaluate(x)
        # val_max = -(val_min - self.f_bias) + self.f_max_target
        return -val_min


class CEC2014F21_Opfunu:
    def __init__(self, n_dimensions):
        self.n_dimensions = n_dimensions
        self.name = "CEC2014F21"
        self.function = F212014(ndim=n_dimensions)

    def evaluate_func(self, x):
        if len(x) != self.n_dimensions:
            raise ValueError(f"CEC2014 F21 function requires {self.n_dimensions} variables, but {len(x)} were given.")
        return self.function.evaluate(x)



class Hypersphere:
    def __init__(self, n_dimensions):
        self.n_dimensions = n_dimensions
        self.name = "Hypersphere"

    def evaluate_func(self, x):
        if len(x) != self.n_dimensions:
            raise ValueError(f"Hypersphere function requires {self.n_dimensions} variables, but {len(x)} were given.")
        return sum([xi ** 2 for xi in x])

class Hyperellipsoid:
    def __init__(self, n_dimensions):
        self.n_dimensions = n_dimensions
        self.name = "Hyperellipsoid"

    def evaluate_func(self, x):
        if len(x) != self.n_dimensions:
            raise ValueError(f"Hyperellipsoid function requires {self.n_dimensions} variables, but {len(x)} were given.")
        return sum([sum([xj ** 2 for xj in x[:i + 1]]) for i in range(self.n_dimensions)])

class EggHolder:
    def __init__(self, n_dimensions):
        self.n_dimensions = n_dimensions
        self.name = "EggHolder"

    def evaluate_func(self, x):
        if len(x) != self.n_dimensions:
            raise ValueError(f"Egg Holder function requires {self.n_dimensions} variables, but {len(x)} were given.")
        return sum([(x[i + 1] + 47) * math.sin(math.sqrt(abs(x[i + 1] + 47 + x[i] / 2))) + x[i] * math.sin(math.sqrt(abs(x[i] - (x[i + 1] + 47)))) for i in range(self.n_dimensions - 1)])

class StyblinskiTang:
    def __init__(self, n_dimensions):
        self.n_dimensions = n_dimensions
        self.name = "StyblinskiTang"

    def evaluate_func(self, x):
        if len(x) != self.n_dimensions:
            raise ValueError(f"Styblinski-Tang function requires {self.n_dimensions} variables, but {len(x)} were given.")
        return sum([xi ** 4 - 16 * xi ** 2 + 5 * xi for xi in x]) / 2

from opfunu.cec_based import F12014

class ReversedCEC2014F1:
    def __init__(self, n_dimensions=10):
        self.n_dimensions = n_dimensions
        self.name = "ReversedCEC2014F1"
        self.function = F12014(ndim=n_dimensions)  # prawdziwa CEC2014 F1

    def evaluate_func(self, x):
        if len(x) != self.n_dimensions:
            raise ValueError(
                f"ReversedCEC2014F1 function requires {self.n_dimensions} variables, "
                f"but {len(x)} were given."
            )

        # Oryginalna wartość CEC2014 F1 (minimum = 100)
        original_value = self.function.evaluate(x)

        # Odwrócenie — teraz maksimum = -100
        return -original_value


class MaximizationExample:
    def __init__(self, n_dimensions=10):
        self.n_dimensions = n_dimensions
        self.name = "MaximizationExample"

    def evaluate_func(self, x):
        if len(x) != self.n_dimensions:
            raise ValueError(f"MaximizationExample function requires {self.n_dimensions} variables, but {len(x)} were given.")
        # Funkcja osiąga maksimum 10, gdy wszystkie wartości x są równe 1
        return 10 - sum([(xi - 1) ** 2 for xi in x])

class GoldsteinAndPrice:
    def __init__(self):
        self.n_dimensions = 2
        self.name = "GoldsteinAndPrice"

    def evaluate_func(self, x):
        if len(x) != self.n_dimensions:
            raise ValueError(f"Goldstein and Price function requires 2 variables, but {len(x)} were given.")
        part1 = (1 + (x[0] + x[1] + 1) ** 2 * (19 - 14 * x[0] + 3 * x[0] ** 2 - 14 * x[1] + 6 * x[0] * x[1] + 3 * x[1] ** 2))
        part2 = (30 + (2 * x[0] - 3 * x[1]) ** 2 * (18 - 32 * x[0] + 12 * x[0] ** 2 + 48 * x[1] - 36 * x[0] * x[1] + 27 * x[1] ** 2))
        return part1 * part2

class FunctionLoader:
    def __init__(self):
        self.folder_path = 'functions_info'
        self.functions = self.load_all_functions()
        self.function_classes = {
            "ackley": Ackley,
            "rastrigin": Rastrigin,
            "rosenbrock": Rosenbrock,
            "sphere": Sphere,
            "griewank": Griewank,
            "schwefel": Schwefel,
            "max": MaximizationExample,
            "michalewicz": Michalewicz,
            "reverse_cec2014_f1": ReversedCEC2014F1,
            "easom": Easom,
            "himmelblau": Himmelblau,
            "keane": Keane,
            "cec2014_f1_opfunu": CEC2014F1_Opfunu,
            "cec2013_f1_opfunu": CEC2013F1_Opfunu,
            "cec2013_f1_opfunu_max": CEC2013F1_OpfunuMAX,
            "cec2014_f21_opfunu": CEC2014F21_Opfunu,
            "rana": Rana,
            "pits_and_holes": PitsAndHoles,
            "hypersphere": Hypersphere,
            "hyperellipsoid": Hyperellipsoid,
            "eggholder": EggHolder,
            "styblinski_tang": StyblinskiTang,
            "goldstein_and_price": GoldsteinAndPrice
        }

    def load_all_functions(self):
        functions = {}
        for file_name in os.listdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), self.folder_path)):
            if file_name.endswith('.json'):
                function_name = file_name.replace('.json', '')
                functions[function_name] = self.load_function_from_json(file_name)
        return functions

    def load_function_from_json(self, file_name):
        file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), self.folder_path, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File '{file_path}' does not exist.")
        with open(file_path, 'r') as file:
            function_data = json.load(file)
        return function_data

    def get_function(self, function_name, n_dimensions):
        if function_name in ["himmelblau", "easom", "pits_and_holes", "goldstein_and_price"]:
            return self.function_classes[function_name]()
        elif function_name in self.function_classes:
            return self.function_classes[function_name](n_dimensions)
        else:
            raise ValueError(f"Function '{function_name}' not found.")

    def evaluate_function(self, function_name, variables, n_dimensions=None):
        if n_dimensions is None:
            n_dimensions = len(variables)
        function_instance = self.get_function(function_name, n_dimensions)
        if len(variables) != n_dimensions:
            raise ValueError(f"Function '{function_name}' requires {n_dimensions} variables, but {len(variables)} were given.")
        return function_instance.evaluate_func(variables)