import time

from opfunu.cec_based import F12014

from detpy.DETAlgs.lshade import LSHADE
from detpy.DETAlgs.data.alg_data import LShadeData
from detpy.models.fitness_function import FitnessFunctionOpfunu
from detpy.models.enums import optimization, boundary_constrain

fitness_fun_opf = FitnessFunctionOpfunu(
    func_type=F12014,
    ndim=10
)

if __name__ == "__main__":
    # Define parameters
    num_of_nfe = 10_000
    population_size = 100
    dimension = 10

    # Define LSHADE parameters using LShadeData
    params_lshade = LShadeData(
        max_nfe=num_of_nfe,
        population_size=population_size,
        dimension=dimension,
        lb=[-100] * dimension,
        ub=[100] * dimension,
        show_plots=True,
        # We want the plots to be rendered (Best fitness per NFE, Average fitness per NFE, Standard deviation per NFE)
        optimization_type=optimization.OptimizationType.MINIMIZATION,
        boundary_constraints_fun=boundary_constrain.BoundaryFixing.RANDOM,
        function=fitness_fun_opf,
        log_population=True,
        parallel_processing=['thread', 1]
    )

    # Initialize and run LSHADE
    start_time = time.time()
    lshade = LSHADE(params_lshade, db_conn='DE.db', db_auto_write=False)
    results = lshade.run()
    elapsed_time = time.time() - start_time

    # Extract and display results
    best_fitness = min(epoch.best_individual.fitness_value for epoch in results.epoch_metrics)
    print(f"Best Fitness Value: {best_fitness}")
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")
