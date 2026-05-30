Installation
============================

To install **DetPy**, use the following command:

.. code-block:: plaintext

   pip install detpy

You can find the full source code on GitHub:
`DetPy Repository <https://github.com/Blazej-Zielinski/detpy/tree/main>`_.

The latest stable release of the library is available on PyPI:
`DetPy on PyPI <https://pypi.org/project/detpy/>`_.

Introduction
============================

DetPy (Differential Evolution Tools) is a library designed to help scientists and engineers solve complex optimization problems using the differential evolution algorithm along with numerous variants.

**Key Features**

- Implementations of popular and state-of-the-art differential evolution methods.
- Flexibility to configure algorithm parameters.
- Visualization tools to monitor results.
- Support for benchmarking against standard optimization functions.
- Option to store results in an SQLite database.

User Guide
============================

Purpose
----------------------------
The goal of DetPy is to simplify the application of differential evolution algorithms for researchers and practitioners in the field of optimization.

Getting Started
----------------------------
To begin using DetPy, follow these steps:

1. **Installation**
   As mentioned above, install the library using ``pip``.

2. **Import the Library**

   .. code-block:: python
      
      import detpy

3. **Basic Usage**

   Here's a quick example of using DetPy to solve a simple optimization problem:

   .. code-block:: python

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
        max_nfe=100,
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

Examples
============================

To make the library more accessible, the documentation includes various examples covering:

- Optimization of the Ackley function based on SADE.
- Optimization of the Ackley function with all DE variants.
- Optimization of common benchmark functions.
- Optimization of functions from Opfunu.

Check out the `Examples Section <https://github.com/Blazej-Zielinski/detpy/tree/main/examples>`_ for more details.
