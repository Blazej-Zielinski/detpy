from detpy.DETAlgs.data.alg_data import MGDEData
from detpy.DETAlgs.mgde import MGDE
from detpy.models.enums.boundary_constrain import BoundaryFixing
from detpy.models.enums.optimization import OptimizationType
from detpy.models.fitness_function import FitnessFunctionOpfunu
from opfunu.cec_based import F82014

if __name__ == "__main__":
    fitness_fun_opf = FitnessFunctionOpfunu(
        func_type=F82014,
        ndim=10
    )

    params = MGDEData(
        epoch=10,
        population_size=10,
        dimension=10,
        lb=[-5, -100, -100, -100, -100, -100, -100, -100, -100, -100],
        ub=[5, 100, 100, 100, 100, 100, 100, 100, 100, 100],
        optimization_type=OptimizationType.MINIMIZATION,
        boundary_constraints_fun=BoundaryFixing.RANDOM,
        function=fitness_fun_opf,
        crossover_rate=0.8,
        log_population=True,
    )
    params.parallel_processing = ['thread', 5]

    default2 = MGDE(params, db_conn="Differential_evolution.db", db_auto_write=False)
    results = default2.run()
    default2.write_results_to_database(results.epoch_metrics)
