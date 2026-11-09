# DetPy (Differential Evolution Tools): A Python toolbox for solving optimization problems using differential evolution

## 📖 Citation

If you use **DetPy** in your research, please cite the relevant publications:

**Version 2.0.0 (2026)**

Groń, K., Golonka, D., & Książek, W. (2026). *Version [2.0.0] - DetPy (Differential evolution tools): A Python toolbox for solving optimization problems using differential evolution.* **SoftwareX**. https://doi.org/10.1016/j.softx.2026.102509

**Original publication (2025)**

Zieliński, B., Ściegienny, S., Orlicki, H., & Książek, W. (2025). *DetPy (Differential Evolution Tools): A Python toolbox for solving optimization problems using differential evolution.* **SoftwareX, 29**, 102014. https://doi.org/10.1016/j.softx.2024.102014

### BibTeX

```bibtex
@article{Gron2026DetPy,
  title   = {Version [2.0.0] - DetPy (Differential evolution tools): A Python toolbox for solving optimization problems using differential evolution},
  author  = {Groń, Konrad and Golonka, Damian and Książek, Wojciech},
  journal = {SoftwareX},
  year    = {2026},
  doi     = {10.1016/j.softx.2026.102509},
}

@article{Zielinski2025DetPy,
  title   = {DetPy (Differential Evolution Tools): A Python toolbox for solving optimization problems using differential evolution},
  author  = {Zieliński, Błażej and Ściegienny, Szymon and Orlicki, Hubert and Książek, Wojciech},
  journal = {SoftwareX},
  volume  = {29},
  pages   = {102014},
  year    = {2025},
  doi     = {10.1016/j.softx.2024.102014}
}
```

# Introduction
The DetPy library contains implementations of the differential evolution algorithm and over 35 variants of this 
algorithm. It can be used to solve advanced optimization problems.
The following variants have been implemented:

| No. | Algorithm                                                                                                                       | Year |
|-----|---------------------------------------------------------------------------------------------------------------------------------|------|
| 1   | DE (Differential evolution) [1]                                                                                                 | 1997 |
| 2   | COMDE (Constrained optimization-based differential evolution) [2]                                                               | 2012 |
| 3   | DERL (Differential evolution random locations) [3]                                                                              | 2006 |
| 4   | NMDE (Novel modified differential evolution algorithm) [4]                                                                      | 2011 |
| 5   | FIADE (Fitness-Adaptive DE) [5]                                                                                                 | 2011 |
| 6   | EMDE (Efficient modified differential evolution) [6]                                                                            | 2015 |
| 7   | IDE (Improved differential evolution) [7]                                                                                       | 2019 |
| 8   | SADE (Self-adaptive differential evolution) [8]                                                                                 | 2008 |
| 9   | JADE (Adaptive differential evolution with optional external archive) [9]                                                       | 2009 |
| 10  | OppBasedDE (Opposition-based differential evolution) [10]                                                                       | 2010 |
| 11  | AADE (Auto adaptive differential evolution algorithm) [11]                                                                      | 2019 |
| 12  | DEGL (Differential evolution with neighborhood-based mutation) [12]                                                             | 2009 |
| 13  | DELB (Differential evolution with localization using the best vector) [3]                                                       | 2006 |
| 14  | EIDE (An efficient improved differential evolution algorithm) [13]                                                              | 2012 |
| 15  | MGDE (A many-objective guided differential evolution) [14]                                                                      | 2022 |
| 16  | ImprovedDE (DE with dynamic mutation parameters) [15]                                                                           | 2023 |
| 17  | SHADE (Success-History Based Parameter Adaptation for Differential Evolution) [16]                                              | 2013 |
| 18  | LSHADE_RSP (Algorithm with a Rank-based Selective Pressure Strategy)  [17]                                                      | 2018 | 
| 19  | LSHADE (Improving the Search Performance of SHADE Using Linear Population Size Reduction) [18]                                  | 2014 |
| 20  | SPS_LSHADE_EIG (Self-Optimizing L-SHADE with Eigenvector Crossover ) [19]                                                       | 2015 |
| 21  | AL-SHADE (Adaptive L-SHADE with current-to-Amean strategy and adaptive mutation selection scheme) [20]                          | 2022 |
| 22  | DETCR (Hybrid DE Algorithm With Adaptive Crossover Operator) [21]                                                               | 2011 |
| 23  | EPSDE (Epsilon Constrained Differential Evolution) [22]                                                                         | 2006 |
| 24  | EPSDEAG (Constrained Optimization by the ε Constrained Differential Evolution with an Archive and Gradient-Based Mutation) [23] | 2010 |
| 25  | LSHADE_EPSIN (L-SHADE with Ensemble Sinusoidal Parameter Adaptation) [24]                                                       | 2016 |
| 26  | EPSDEG (Epsilon Constrained Differential Evolution with Gradient-Based Mutation) [25]                                           | 2009 |
| 27  | EPSADE (Epsilon Constrained adaptive Differential Evolution) [26]                                                               | 2010 |
| 28  | EPSRDE (Epsilon Constrained Rank-Based Differential Evolution) [27]                                                             | 2012 |
| 29  | SHADE 1.1 (Success-History Based Parameter Adaptation for Differential Evolution) [16]                                          | 2014 |
| 30  | EPSDEwDC (Epsilon Constrained Differential Evolution with Dynamic ε-Level Control) [28]                                         | 2012 |
| 31  | SHADE4 Evaluating the Performance of SHADE with Competing Strategies on CEC 2014 Single-Parameter Test Suit [29]                | 2016 |
| 32  | LSHADE44 L-SHADE with Competing Strategies Applied to CEC2015 Learning-based Test Suite [30]                                    | 2016 |
| 33  | FDDE (Fitness-Diversity Differential Evolution) [31]                                                                            | 2021 |
| 34  | iL-SHADE (Improved L-SHADE Algorithm for Single Objective Real-Parameter Optimization) [32]                                     | 2016 |
| 35  | jSO (Single Objective Real-Parameter Optimization: Algorithm jSO) [33]                                                          | 2017 |
| 36  | D-SHADE (L-SHADE with dynamic population size reduction) [18, 34]                                                               | 2014 |
| 37  | MjSO (Modified jSO with probability selection and directed mutation) [35]                                                       | 2023 |
| 38  | jSOa (Progressive Archive in Adaptive jSO Algorithm) [36]                                                                       | 2024 |
| 39  | jSOe (Eigenvector Crossover in the Efficient jSO Algorithm) [37]                                                                | 2019 |

# Installation
```bash
pip install detpy
```

### Optional dependencies

For gradient-based constraint handling (EpsDEAG with automatic/symbolic derivatives):
```bash
pip install "detpy[gradient]"
```

For TensorBoard monitoring:
```bash
pip install "detpy[tensorboard]"
```

All optional dependencies at once:
```bash
pip install "detpy[all]"
```

# Quick Start

```python
from detpy.DETAlgs.data.alg_data import DEData
from detpy.DETAlgs.de import DE
from detpy.functions import FunctionLoader
from detpy.models.enums.boundary_constrain import BoundaryFixing
from detpy.models.enums.optimization import OptimizationType
from detpy.models.fitness_function import BenchmarkFitnessFunction

# Load a benchmark function
function_loader = FunctionLoader()
ackley = function_loader.get_function(function_name="ackley", n_dimensions=2)

# Set up and run optimization
params = DEData(
    max_nfe=10000,
    population_size=50,
    dimension=2,
    lb=[-32.768, -32.768],
    ub=[32.768, 32.768],
    optimization_type=OptimizationType.MINIMIZATION,
    boundary_constraints_fun=BoundaryFixing.RANDOM,
    function=BenchmarkFitnessFunction(ackley),
)

algorithm = DE(params)
results = algorithm.run()
print(f"Best fitness: {results.best_fitness}")
```

# Example — SADE with Database and Parallel Processing

```python
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
    max_nfe=10000,
    population_size=100,
    dimension=2,
    lb=[-32.768, -32.768],
    ub=[32.768, 32.768],
    optimization_type=OptimizationType.MINIMIZATION,
    boundary_constraints_fun=BoundaryFixing.RANDOM,
    function=fitness_fun,
    log_population=True,
    parallel_processing=['thread', 4]
)

sade = SADE(params, db_conn="results.db", db_auto_write=False)
results = sade.run()
```

# Using FunctionLoader

DetPy includes a set of built-in benchmark functions. Use `FunctionLoader` to load a function
by name and specify the number of dimensions:

```python
from detpy.functions import FunctionLoader

function_loader = FunctionLoader()

# Load a function and evaluate it
ackley = function_loader.get_function(function_name="ackley", n_dimensions=2)
result = function_loader.evaluate_function("ackley", [0.0, 0.0], n_dimensions=2)
```

Available functions:

| Function | Name |
|----------|------|
| Ackley | `ackley` |
| Rastrigin | `rastrigin` |
| Rosenbrock | `rosenbrock` |
| Sphere | `sphere` |
| Griewank | `griewank` |
| Schwefel | `schwefel` |
| Michalewicz | `michalewicz` |
| Easom | `easom` |
| Himmelblau | `himmelblau` |
| Keane | `keane` |
| Rana | `rana` |
| Pits and Holes | `pits_and_holes` |
| Hypersphere | `hypersphere` |
| Hyperellipsoid | `hyperellipsoid` |
| Egg Holder | `eggholder` |
| Styblinski-Tang | `styblinski_tang` |
| Goldstein-Price | `goldstein_and_price` |

**Constrained benchmark functions (G-functions):**

| Function | Name |
|----------|------|
| G01 – G12 | `G01`, `G02`, ..., `G12` |

Test functions prepared based on https://gitlab.com/luca.baronti/python_benchmark_functions

# Using Opfunu Functions

DetPy integrates with [Opfunu](https://github.com/thieu1995/opfunu), providing access to
CEC competition benchmark suites (CEC2014, CEC2017, etc.):

```python
from opfunu.cec_based import F82014
from detpy.DETAlgs.data.alg_data import LShadeData
from detpy.DETAlgs.lshade import LSHADE
from detpy.models.enums.boundary_constrain import BoundaryFixing
from detpy.models.enums.optimization import OptimizationType
from detpy.models.fitness_function import FitnessFunctionOpfunu

fitness_fun = FitnessFunctionOpfunu(func_type=F82014, ndim=10)

params = LShadeData(
    max_nfe=100000,
    population_size=100,
    dimension=10,
    lb=[-100] * 10,
    ub=[100] * 10,
    optimization_type=OptimizationType.MINIMIZATION,
    boundary_constraints_fun=BoundaryFixing.RANDOM,
    function=fitness_fun,
    log_population=True,
)

lshade = LSHADE(params)
results = lshade.run()
```

# Monitoring

DetPy supports real-time monitoring of optimization progress via CSV logs or TensorBoard.

### CSV Monitoring

```python
from detpy.monitoring.csv_monitor import CSVMonitor

monitor = CSVMonitor(
    log_dir="results",
    experiment_name="my_experiment",
    every_nfe=100,
    flush_every=1
)

algorithm = LSHADE(params, monitor=monitor)
results = algorithm.run()
```

This creates two CSV files in the output directory: `general_metrics.csv` and `specific_metrics.csv`.

### TensorBoard Monitoring

First, install the optional TensorBoard dependency:

```bash
pip install "detpy[tensorboard]"
```

Then use `TensorBoardMonitor`:

```python
from detpy.monitoring import TensorBoardMonitor

monitor = TensorBoardMonitor(
    log_dir="tb_logs",
    experiment_name="run_1",
    flush_every=10,
    every_nfe=1000
)

algorithm = LSHADE(params, monitor=monitor)
results = algorithm.run()
```

Launch the TensorBoard dashboard with:

```bash
tensorboard --logdir=tb_logs
```

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
20. Yintong Li, Tong Han, Huan Zhou, Shangqin Tang, Hui Zhao. *A novel adaptive L-SHADE algorithm and its application in UAV swarm resource configuration problem*. Information Sciences, vol. 606, pp. 350–367, 2022.
21. Gilberto Reynoso-Meza, Javier Sanchis, Xavier Blasco, Juan M. Herrero, *Hybrid DE algorithm with adaptive crossover operator for solving real-world numerical optimization problems*. IEEE Congress of Evolutionary Computation (CEC), New Orleans, LA, USA, 2011, doi: 10.1109/CEC.2011.5949800, 2011.
22. Tetsuyuki Takahama, Setsuko Sakai, Noriyuki Iwane, *Solving Nonlinear Constrained Optimization Problems by the Epsilon Constrained Differential Evolution* IEEE International Conference on Systems, Man and Cybernetics, 2006.
23. Tetsuyuki Takahama, Setsuko Sakai, *Constrained optimization by the ε constrained differential evolution with an archive and gradient-based mutation*. IEEE Congress on Evolutionary Computation, 2010.
24. Noor H. Awad, Mostafa Z. Ali, Ponnuthurai N. Suganthan, Robert G. Reynolds. *L-SHADE with Ensemble Sinusoidal Parameter Adaptation*. IEEE Congress on Evolutionary Computation, 2016.
25. T. Takahama and S. Sakai, *Solving difficult constrained optimization problems by the ε constrained differential evolution with gradient based mutation*, in Constraint-Handling in Evolutionary Optimization, E. Mezura-Montes, Ed. Springer-Verlag, pp. 51–72, 2009
26. Tetsuyuki Takahama and Setsuko Sakai, *Efficient Constrained Optimization by the ε Constrained Adaptive Differential Evolution*. IEEE Congress on Evolutionary Computation, 2010
27. Tetsuyuki Takahama and Setsuko Sakai, *Efficient Constrained Optimization by the ε Constrained Rank-Based Differential Evolution*. IEEE Congress on Evolutionary Computation, 2012
28. Tetsuyuki Takahama and Setsuko Sakai, *Epsilon Constrained Differential Evolution with Dynamic ε-Level Control*. Chakraborty, U.K. (eds) Advances in Differential Evolution. Studies in Computational Intelligence, vol 143. Springer, Berlin, Heidelberg, 2008
29. Petr Bujok, Josef Tvrdík, Radka Poláková *Evaluating the performance of SHADE with competing strategies on CEC 2014 single-parameter test suite* IEEE Congress on Evolutionary Computation, 2016 
30. Petr Bujok, Josef Tvrdík, Radka Poláková *L-SHADE with Competing Strategies Applied to CEC2015 Learning-based Test Suite* IEEE Congress on Evolutionary Computation, 2016 
31. Cheng, J., Pan, Z., Liang, H., Gao, Z., Gao, J. *Differential evolution algorithm with fitness and diversity ranking-based mutation operator*. Swarm and Evolutionary Computation, vol. 61, 100816, 2021.
32. Janez Brest, Mirjam Sepesy Maučec, Borko Boškovič. *iL-SHADE: Improved L-SHADE algorithm for single objective real-parameter optimization*. IEEE Congress on Evolutionary Computation (CEC), 2016.
33. Janez Brest, Mirjam Sepesy Maučec, Borko Boškovič. *Single objective real-parameter optimization: Algorithm jSO*. IEEE Congress on Evolutionary Computation (CEC), 2017.
34. D-SHADE combines L-SHADE [18] with Janez Brest, Mirjam Sepesy Maučec. *Population size reduction for the differential evolution algorithm*. Applied Intelligence, vol. 29, no. 3, pp. 228–247, 2008. https://doi.org/10.1007/s10489-007-0091-x
35. Yintong Li, Tong Han, Xiaofei Wang, Huan Zhou, Shangqin Tang, Changqiang Huang, Bo Han. *MjSO: A modified differential evolution with a probability selection mechanism and a directed mutation strategy*. Swarm and Evolutionary Computation, vol. 78, 101294, 2023. https://doi.org/10.1016/j.swevo.2023.101294
36. Petr Bujok. *Progressive archive in adaptive jSO algorithm*. Mathematics, vol. 12, no. 16, 2534, 2024. https://doi.org/10.3390/math12162534
37. Petr Bujok, Radka Poláková. *Eigenvector crossover in the efficient jSO algorithm*. MENDEL Soft Computing Journal, vol. 25, no. 1, pp. 65–72, 2019. https://doi.org/10.13164/mendel.2019.1.065
# Documentation
Full documentation is available: https://blazej-zielinski.github.io/detpy/

