import time

from opfunu.cec_based import F12014

from detpy.DETAlgs.lshade import LSHADE
from detpy.DETAlgs.data.alg_data import LShadeData
from detpy.models.fitness_function import FitnessFunctionOpfunu
from detpy.models.enums import optimization, boundary_constrain
from detpy.models.stop_condition.stop_condition import StopCondition


class StopWhenFitnessBelowThreshold(StopCondition):
    def should_stop(self, nfe, epoch, best_chromosome):
        # Stop if the best fitness value is less than 100,000
        # The parameters `nfe` (Number of Function Evaluations), `epoch` (current iteration number),
        # and `best_chromosome` (the best solution found so far) can be used to define the stopping criterion.
        return best_chromosome.fitness_value < 100_000


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
        show_plots=False,
        additional_stop_criteria=StopWhenFitnessBelowThreshold(),
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
