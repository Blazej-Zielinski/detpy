# DetPy (Differential Evolution Tools): A Python toolbox for solving optimization problems using differential evolution

# Introduction
The DetPy library contains implementations of the differential evolution algorithm and 18 modifications of this 
algorithm. It can be used to solve advanced optimization problems.
The following variants have been implemented:

| No. | Algorithm                                                                                     | Year |
|-----|-----------------------------------------------------------------------------------------------|------|
| 1   | DE (Differential evolution) [1]                                                               | 1997 |
| 2   | COMDE (Constrained optimization-based differential evolution) [2]                             | 2012 |
| 3   | DERL (Differential evolution random locations) [3]                                            | 2006 |
| 4   | NMDE (Novel modified differential evolution algorithm) [4]                                    | 2011 |
| 5   | FIADE (Fitness-Adaptive DE) [5]                                                               | 2011 |
| 6   | EMDE (Efficient modified differential evolution) [6]                                          | 2015 |
| 7   | IDE (Improved differential evolution) [7]                                                     | 2019 |
| 8   | SADE (Self-adaptive differential evolution) [8]                                               | 2008 |
| 9   | JADE (Adaptive differential evolution with optional external archive) [9]                     | 2009 |
| 10  | OppBasedDE (Opposition-based differential evolution) [10]                                     | 2010 |
| 11  | AADE (Auto adaptive differential evolution algorithm) [11]                                    | 2019 |
| 12  | DEGL (Differential evolution with neighborhood-based mutation) [12]                           | 2009 |
| 13  | DELB (Differential evolution with localization using the best vector) [3]                     | 2006 |
| 14  | EIDE (An efficient improved differential evolution algorithm) [13]                            | 2012 |
| 15  | MGDE (A many-objective guided differential evolution) [14]                                    | 2022 |
| 16  | ImprovedDE (DE with dynamic mutation parameters) [15]                                         | 2023 |
| 17  | SHADE (Success-History Based Parameter Adaptation for Differential Evolution) [16]            | 2013 |
| 18  | LSHADE_RSP (Algorithm with a Rank-based Selective Pressure Strategy)  [17]                    | 2018 | 
| 19  | LSHADE (Improving the Search Performance of SHADE Using Linear Population Size Reduction) [18]| 2014 |
| 20  | SPS_LSHADE_EIG (Self-Optimizing L-SHADE with Eigenvector Crossover ) [19]                     | 2015 |

# Installation
```
pip install detpy
```

# Example - optimization of the Ackley function based SADE
```
from detpy.DETAlgs.data.alg_data import SADEData

from detpy.DETAlgs.sade import SADE

from detpy.functions import FunctionLoader

from detpy.models.enums.boundary_constrain import BoundaryFixing

from detpy.models.enums.optimization import OptimizationType

from detpy.models.fitness_function import BenchmarkFitnessFunction


function_loader = FunctionLoader()

ackley_function = function_loader.get_function(function_name="ackley", n_dimensions=2)

fitness_fun = BenchmarkFitnessFunction(ackley_function)


params = SADEData(

    epoch=100,

    population_size=100,

    dimension=2,

    lb=[-32.768, -32.768],

    ub=[32.768, 32.768],

    mode=OptimizationType.MINIMIZATION,

    boundary_constraints_fun=BoundaryFixing.RANDOM,

    function=fitness_fun,

    log_population=True,

    parallel_processing=['thread', 4]

)


default2 = SADE(params, db_conn="Differential_evolution.db", db_auto_write=False)

results = default2.run()
```

# Using FunctionLoader

You can also use one of predefined functions to solve your problem. 
To do this, call the FunctionLoader method and pass as an argument the name of a function from the folder and variables,
which u want to use in your calculations.

```
function_loader = FunctionLoader()
function_name = "ackley"
variables = [0.0, 0.0]
n_dimensions = 2

result = function_loader.evaluate_function(function_name, variables, n_dimensions)
```

Available functions:

```
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
            "goldstein_and_price": GoldsteinAndPrice
        }
```

Test functions prepared based on https://gitlab.com/luca.baronti/python_benchmark_functions

# References

1. Storn, Rainer and Price, Kenneth. *Differential Evolution - A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces*. Journal of Global Optimization, vol. 11, no. 4, 1997.
2. Mohamed, Ali Wagdy and Sabry, Hegazy Zaher. *Constrained optimization based on modified differential evolution algorithm*. Information Sciences, vol. 194, 2012.
3. Kaelo, Paul and Ali, Mohamed M. *A numerical study of some modified differential evolution algorithms*. European Journal of Operational Research, vol. 169, no. 3, 2006.
4. Zou, Dexuan, Liu, Haikuan, Gao, Liqun, Li, Steven. *A novel modified differential evolution algorithm for constrained optimization problems*. Computers & Mathematics with Applications, vol. 61, no. 6, 2011.
5. Ghosh, Arnob, Das, Swagatam, Chowdhury, Aritra, Giri, Ritwik. *An improved differential evolution algorithm with fitness-based adaptation of the control parameters*. Information Sciences, vol. 181, no. 18, 2011.
6. Mohamed, Ali Wagdy. *An efficient modified differential evolution algorithm for solving constrained non-linear integer and mixed-integer global optimization problems*. International Journal of Machine Learning and Cybernetics, vol. 8, no. 3, 2015.
7. Ma, Jian and Li, Haiming. *Research on Rosenbrock Function Optimization Problem Based on Improved Differential Evolution Algorithm*. Journal of Computer and Communications, vol. 7, no. 11, 2019.
8. Wu Zhi-Feng, Huang Hou-Kuan, Yang Bei, Zhang Ying. *A modified differential evolution algorithm with self-adaptive control parameters*. 2008 3rd International Conference on Intelligent System and Knowledge Engineering, IEEE, 2008.
9. Zhang, Jingqiao and Sanderson, A.C. *JADE: Adaptive Differential Evolution With Optional External Archive*. IEEE Transactions on Evolutionary Computation, vol. 13, no. 5, 2009.
10. Rahnamayan, Shahryar, Tizhoosh, Hamid R., Salama, Magdy M. A. *Opposition-Based Differential Evolution*. Studies in Computational Intelligence, Springer, Berlin, Heidelberg.
11. Sharma, Vivek, Agarwal, Shalini, Verma, Pawan Kumar. *Auto Adaptive Differential Evolution Algorithm*. 2019 3rd International Conference on Computing Methodologies and Communication (ICCMC), IEEE, 2019.
12. Das, Swagatam, Abraham, Ajith, Chakraborty, Uday K., Konar, Amit. *Differential Evolution Using a Neighborhood-Based Mutation Operator*. IEEE Transactions on Evolutionary Computation, vol. 13, no. 3, 2009.
13. Zou, Dexuan and Gao, Liqun. *An efficient improved differential evolution algorithm*. Proceedings of the 31st Chinese Control Conference, 2012.
14. Zouache, Djaafar, Abdelaziz, Fouad Ben. *MGDE: a many-objective guided differential evolution with strengthened dominance relation and bi-goal evolution*. Annals of Operations Research, Springer, 2022.
15. Lin, Yifeng, Yang, Yuer, Zhang, Yinyan. *Improved differential evolution with dynamic mutation parameters*. Soft Computing, Springer, 2023.
16. Ryoji Tanabe, Alex Fukunaga. *Success-history based parameter adaptation for Differential Evolution*. IEEE Congress on Evolutionary Computation, 2013.
17. Vladimir Stanovov, Shakhnaz Akhmedova, Eugene Semenkin. *Algorithm with a Rank-based Selective Pressure mutation (LSAHDE-RSP)*. IEEE Congress on Evolutionary Computation, 2018.
18. Ryoji Tanabe, Alex Fukunaga. *Improving the Search Performance of SHADE Using Linear Population Size Reduction*. IEEE Congress on Evolutionary Computation, 2014.
19. Shu-Mei Guo, Jason Sheng-Hong Tsai, Chin-Chang Yang, Pang-Han Hsu. *A self-optimization approach for L-SHADE incorporated with eigenvector-based crossover and successful-parent-selecting framework on CEC 2015 benchmark set*. IEEE Congress on Evolutionary Computation (CEC), 2015.

# Documentation
Full documentation is available: https://blazej-zielinski.github.io/detpy/

