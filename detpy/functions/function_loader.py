import os
import json
import math
import numpy as np
from scipy.stats import multivariate_normal
from abc import ABC, abstractmethod
from detpy.models.enums import optimization


class Ackley:
    def __init__(self, n_dimensions):
        self.n_dimensions = n_dimensions
        self.name = "Ackley"

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
            raise ValueError(
                f"Hyperellipsoid function requires {self.n_dimensions} variables, but {len(x)} were given.")
        return sum([sum([xj ** 2 for xj in x[:i + 1]]) for i in range(self.n_dimensions)])


class EggHolder:
    def __init__(self, n_dimensions):
        self.n_dimensions = n_dimensions
        self.name = "EggHolder"

    def evaluate_func(self, x):
        if len(x) != self.n_dimensions:
            raise ValueError(f"Egg Holder function requires {self.n_dimensions} variables, but {len(x)} were given.")
        return sum([(x[i + 1] + 47) * math.sin(math.sqrt(abs(x[i + 1] + 47 + x[i] / 2))) + x[i] * math.sin(
            math.sqrt(abs(x[i] - (x[i + 1] + 47)))) for i in range(self.n_dimensions - 1)])


class StyblinskiTang:
    def __init__(self, n_dimensions):
        self.n_dimensions = n_dimensions
        self.name = "StyblinskiTang"

    def evaluate_func(self, x):
        if len(x) != self.n_dimensions:
            raise ValueError(
                f"Styblinski-Tang function requires {self.n_dimensions} variables, but {len(x)} were given.")
        return sum([xi ** 4 - 16 * xi ** 2 + 5 * xi for xi in x]) / 2

class GoldsteinAndPrice:
    def __init__(self):
        self.n_dimensions = 2
        self.name = "GoldsteinAndPrice"

    def evaluate_func(self, x):
        if len(x) != self.n_dimensions:
            raise ValueError(f"Goldstein and Price function requires 2 variables, but {len(x)} were given.")
        part1 = (1 + (x[0] + x[1] + 1) ** 2 * (
                19 - 14 * x[0] + 3 * x[0] ** 2 - 14 * x[1] + 6 * x[0] * x[1] + 3 * x[1] ** 2))
        part2 = (30 + (2 * x[0] - 3 * x[1]) ** 2 * (
                18 - 32 * x[0] + 12 * x[0] ** 2 + 48 * x[1] - 36 * x[0] * x[1] + 27 * x[1] ** 2))
        return part1 * part2

class ConstrainedBenchmark(ABC):

    def __init__(self, name, n_dimensions, lower_bounds, upper_bounds, h, g, optimization_type):
        self.n_dimensions = n_dimensions
        self.name = name
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.h = h
        self.g = g
        self.optimization_type = optimization_type

    @abstractmethod
    def evaluate_func(self, x):
        pass


class G01(ConstrainedBenchmark):
    def evaluate_func(self, x):
        part1 = 5 * sum(x[0:4])
        part2 = -5 * sum([xi ** 2 for xi in x[0:4]])
        part3 = -sum(x[4:13])
        return part1 + part2 + part3

    def __init__(self, n_dimensions):
        n_dimensions = 13
        super().__init__(
            name = "G01",
            n_dimensions = n_dimensions,
            lower_bounds = [0.0] * n_dimensions,
            upper_bounds = [1.0] * 9 + [100.0] * 3 + [1.0],
            optimization_type= optimization.OptimizationType.MINIMIZATION,
            h = [],
            g = [
            lambda x: 2 * x[0] + 2 * x[1] + x[9] + x[10] - 10,
            lambda x: 2 * x[0] + 2 * x[2] + x[9] + x[11] - 10,
            lambda x: 2 * x[1] + 2 * x[2] + x[10] + x[11] - 10,
            lambda x: -8 * x[0] + x[9],
            lambda x: -8 * x[1] + x[10],
            lambda x: -8 * x[2] + x[11],
            lambda x: -2 * x[3] - x[4] + x[9],
            lambda x: -2 * x[5] - x[6] + x[10],
            lambda x: -2 * x[7] - x[12] + x[11],
        ]
        )

class G02(ConstrainedBenchmark):
    def evaluate_func(self, x):
        num = np.sum(np.cos(x) ** 4) - 2.0 * np.prod(np.cos(x) ** 2)
        denom = math.sqrt(np.sum((np.arange(1, len(x) + 1)) * (np.pow(x, 2))))
        return abs(num / denom)

    def __init__(self, n_dimensions):
        n_dimensions = 20
        super().__init__(
            name="G02",
            n_dimensions=n_dimensions,
            lower_bounds=[0.0] * n_dimensions,
            upper_bounds=[10.0] * n_dimensions,
            optimization_type=optimization.OptimizationType.MAXIMIZATION,
            h=[],
            g=[
                lambda x : np.sum(x) - 7.5 * n_dimensions,
                lambda x : 0 - (np.prod(x) - 0.75),
                ]
        )

class G03(ConstrainedBenchmark):
    def evaluate_func(self, x):
        return (math.sqrt(self.n_dimensions) ** self.n_dimensions) * np.prod(x)

    def __init__(self, n_dimensions):
        n_dimensions = 10
        super().__init__(
            name="G03",
            n_dimensions=n_dimensions,
            lower_bounds=[0.0] * n_dimensions,
            upper_bounds=[1.0] * n_dimensions,
            optimization_type=optimization.OptimizationType.MAXIMIZATION,
            h=[
                lambda x: np.sum(np.pow(x, 2)) - 1,
            ],
            g=[]
        )


class G04(ConstrainedBenchmark):
    def evaluate_func(self, x):
        return  5.3578547 * x[2]**2 + 0.8356891 * x[0]*x[4] + 37.293239 * x[0] - 40792.141

    def __init__(self, n_dimensions):
        super().__init__(
            name="G04",
            n_dimensions=5,
            lower_bounds=[78.0, 33.0, 27.0, 27.0, 27.0],
            upper_bounds=[102.0, 45.0, 45.0, 45.0, 45.0],
            optimization_type=optimization.OptimizationType.MINIMIZATION,
            h=[],
            g=[
                lambda x: 85.334407 + 0.0056858 * x[1] * x[4] + 0.0006262 * x[0] * x[3] - 0.0022053 * x[2] * x[4] - 92,
                lambda x: 0 - (85.334407 + 0.0056858 * x[1] * x[4] + 0.0006262 * x[0] * x[3] - 0.0022053 * x[2] * x[4]),
                lambda x: 80.51249 + 0.0071317 * x[1] * x[4] + 0.0029955 * x[0] * x[1] + 0.0021813 * x[2] ** 2 - 110,
                lambda x: 90 - (80.51249 + 0.0071317 * x[1] * x[4] + 0.0029955 * x[0] * x[1] + 0.0021813 * x[2] ** 2),
                lambda x: 9.300961 + 0.0047026 * x[2] * x[4] + 0.0012547 * x[0] * x[2] + 0.0019085 * x[2] * x[3] - 25,
                lambda x: 20 - (9.300961 + 0.0047026 * x[2] * x[4] + 0.0012547 * x[0] * x[2] + 0.0019085 * x[2] * x[3]),
            ]
        )

class G05(ConstrainedBenchmark):
    def evaluate_func(self, x):
        return 3 * x[0] + 0.000001 * x[0] ** 3 + 2 * x[1] + 0.000002 / 3 * x[1] ** 3

    def __init__(self, n_dimensions):
        super().__init__(
            name="G05",
            n_dimensions=4,
            lower_bounds=[0.0, 0.0, -0.55, -0.55],
            upper_bounds=[1200, 1200, 0.55, 0.55],
            optimization_type=optimization.OptimizationType.MINIMIZATION,
            h=[
                lambda x: 1000 * np.sin(-x[2] - 0.25) + 1000 * np.sin(-x[3] - 0.25) + 894.8 - x[0],
                lambda x: 1000 * np.sin(x[2] - 0.25) + 1000 * np.sin(x[2] - x[3] - 0.25) + 894.8 - x[1],
                lambda x: 1000 * np.sin(x[3] - 0.25) + 1000 * np.sin(x[3] - x[2] - 0.25) + 1294.8],
            g=[
                lambda x: 0 - (x[3] - x[2] + 0.55),
                lambda x: 0 - (x[2] - x[3] + 0.55)
            ]
        )

class G06(ConstrainedBenchmark):
    def evaluate_func(self, x):
        return (x[0] - 10) ** 3 + (x[1] - 20) ** 3

    def __init__(self, n_dimensions):
        super().__init__(
            name="G06",
            n_dimensions=2,
            lower_bounds=[13.0, 0.0],
            upper_bounds=[100.0, 100.0],
            optimization_type=optimization.OptimizationType.MINIMIZATION,
            h=[],
            g=[
                lambda x: 0 - ((x[0] - 5) ** 2 + (x[1] - 5) ** 2 - 100),
                lambda x: 0 - (-(x[0] - 6) ** 2 - (x[1] - 5) ** 2 + 82.81),
            ]
        )

class G07(ConstrainedBenchmark):
    def evaluate_func(self, x):
        return (
        x[0]**2 + x[1]**2 + x[0]*x[1] - 14*x[0] - 16*x[1]
        + (x[2] - 10)**2
        + 4*(x[3] - 5)**2
        + (x[4] - 3)**2
        + 2*(x[5] - 1)**2
        + 5*x[6]**2
        + 7*(x[7] - 11)**2
        + 2*(x[8] - 10)**2
        + (x[9] - 7)**2
        + 45
    )

    def __init__(self, n_dimensions):
        n_dimensions = 10
        super().__init__(
            name="G07",
            n_dimensions=n_dimensions,
            lower_bounds=[-10.0] * n_dimensions,
            upper_bounds=[10.0] * n_dimensions,
            optimization_type=optimization.OptimizationType.MINIMIZATION,
            h=[],
            g=[
                lambda x: 0 - (105 - 4 * x[0] - 5 * x[1] + 3 * x[6] - 9 * x[7]),
                lambda x: 0 - (-3 * (x[0] - 2) ** 2 - 4 * (x[1] - 3) ** 2 - 2 * x[2] ** 2 + 7 * x[3] + 120),
                lambda x: 0 - (-10 * x[0] + 8 * x[1] + 17 * x[6] - 2 * x[7]),
                lambda x: 0 - (- x[0] ** 2 - 2 * (x[1] - 2) ** 2 + 2 * x[0] * x[1] - 14 * x[4] + 6 * x[5]),
                lambda x: 0 - (8 * x[0] - 2 * x[1] - 5 * x[8] + 2 * x[9] + 12),
                lambda x: 0 - (-5 * x[0] ** 2 - 8 * x[1] - (x[2] - 6) ** 2 + 2 * x[3] + 40),
                lambda x: 0 - (3 * x[0] - 6 * x[1] - 12 * (x[8] - 8) ** 2 + 7 * x[9]),
                lambda x: 0 - (-0.5 * (x[0] - 8) ** 2 + 2 * (x[1] - 4) ** 2 + 3 * x[4] ** 2 + x[5] + 30)
            ]
        )

class G08(ConstrainedBenchmark):
    def evaluate_func(self, x):
         return (np.sin(2 * np.pi * x[0]) ** 3 * np.sin(2 * np.pi * x[1])) / (x[0] ** 3 * (x[0] + x[1]))

    def __init__(self, n_dimensions):
        n_dimensions = 2
        super().__init__(
            name="G08",
            n_dimensions=n_dimensions,
            lower_bounds=[1e-8] * n_dimensions,
            upper_bounds=[10.0] * n_dimensions,
            optimization_type=optimization.OptimizationType.MAXIMIZATION,
            h=[],
            g=[
                lambda x: x[0] ** 2 - x[1] + 1,
                lambda x: 1 - x[0] + (x[1] - 4) ** 2,
            ]
        )

class G09(ConstrainedBenchmark):
    def evaluate_func(self, x):
        return (x[0] - 10) ** 2 + 5 * (x[1] - 12) ** 2 + x[2] ** 4 + 3 * (x[3] - 11) ** 2 + 10 *x[4] ** 6 + 7 * x[5] **  2 + x[6] ** 4 - 4 * x[5] * x[6] - 10 * x[5] - 8 * x[6]

    def __init__(self, n_dimensions):
        n_dimensions = 7
        super().__init__(
            name="G09",
            n_dimensions=n_dimensions,
            lower_bounds=[-10.0] * n_dimensions,
            upper_bounds=[10.0] * n_dimensions,
            optimization_type=optimization.OptimizationType.MINIMIZATION,
            h=[],
            g=[
                lambda x: 0 - (127 - 2 * x[0] ** 2 - 3 * x[1] ** 4 - x[2] - 4 * x[3] ** 2 - 5 * x[4]),
                lambda x: 0 - (282 - 7 * x[0] - 3 * x[1] - 10 * x[2] ** 2 - x[3] + x[4]),
                lambda x: 0 - (196 - 23 * x[0] - x[1] ** 2 - 6 * x[5] ** 2 + 8 * x[6]),
                lambda x: 0 - (-4 * x[0] ** 2 - x[1] ** 2 + 3 * x[0] * x[1] - 2 * x[2] ** 2 - 5 * x[5] + 11 * x[6]),
            ]
        )

class G10(ConstrainedBenchmark):
    def evaluate_func(self, x):
        return x[0] + x[1] + x[2]

    def __init__(self, n_dimensions):
        super().__init__(
            name="G10",
            n_dimensions=8,
            lower_bounds=[100.0, 1000.0, 1000.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            upper_bounds=[10000.0, 10000.0, 10000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0],
            optimization_type=optimization.OptimizationType.MINIMIZATION,
            h=[],
            g=[
                lambda x: 0 - (1 - 0.0025 * (x[3] + x[5])),
                lambda x: 0 - (1 - 0.0025 * (x[4] + x[6] - x[3])),
                lambda x: 0 - (1 - 0.01 * (x[7] - x[4])),
                lambda x: 0 - (x[0] * x[5] - 833.33252 * x[3] - 100 * x[0] + 83333.333),
                lambda x: 0 - (x[1] * x[6] - 1250 * x[4] - x[1] * x[3] + 1250 * x[3]),
                lambda x: 0 - (x[2] * x[7] - 1250000 - x[2] * x[4] + 2500 * x[4]),
            ]
        )

class G11(ConstrainedBenchmark):
    def evaluate_func(self, x):
        return x[0] ** 2 + (x[1] - 1) ** 2

    def __init__(self, n_dimensions):
        n_dimensions = 2
        super().__init__(
            name="G11",
            n_dimensions=n_dimensions,
            lower_bounds=[-1.0] * n_dimensions,
            upper_bounds=[1.0] * n_dimensions,
            optimization_type=optimization.OptimizationType.MINIMIZATION,
            h=[
                lambda x: x[1] - x[0] ** 2
            ],
            g=[]
        )

class G12(ConstrainedBenchmark):
    def evaluate_func(self, x):
        return -(100 - (x[0] - 5) ** 2 - (x[1] - 5) ** 2 - (x[2] - 5) ** 2) / 100

    def __init__(self, n_dimensions):
        n_dimensions = 3
        super().__init__(
            name="G12",
            n_dimensions=n_dimensions,
            lower_bounds=[0.0] * n_dimensions,
            upper_bounds=[10.0] * n_dimensions,
            optimization_type=optimization.OptimizationType.MINIMIZATION,
            h=[],
            g=[
                lambda x: (x[0] - 5) ** 2 + (x[1] - 5) ** 2 + (x[2] - 5) ** 2 - 0.0625
            ]
        )

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
            "michalewicz": Michalewicz,
            "easom": Easom,
            "himmelblau": Himmelblau,
            "keane": Keane,
            "rana": Rana,
            "pits_and_holes": PitsAndHoles,
            "hypersphere": Hypersphere,
            "hyperellipsoid": Hyperellipsoid,
            "eggholder": EggHolder,
            "styblinski_tang": StyblinskiTang,
            "G01": G01,
            "G02": G02,
            "G03": G03,
            "G04": G04,
            "G05": G05,
            "G06": G06,
            "G07": G07,
            "G08": G08,
            "G09": G09,
            "G10": G10,
            "G11": G11,
            "G12": G12,
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
            raise ValueError(
                f"Function '{function_name}' requires {n_dimensions} variables, but {len(variables)} were given.")
        return function_instance.evaluate_func(variables)
