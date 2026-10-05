import time

from opfunu.cec_based import F12014

from detpy.DETAlgs.data.alg_data import LSHADE44Data
from detpy.DETAlgs.lshade44 import LSHADE44
from detpy.models.enums import optimization, boundary_constrain
from detpy.models.fitness_function import FitnessFunctionOpfunu
from detpy.monitoring import TensorBoardMonitor
from detpy.monitoring.csv_monitor import CSVMonitor

fitness_fun_opf = FitnessFunctionOpfunu(
    func_type=F12014,
    ndim=10
)

if __name__ == "__main__":
    num_of_nfe = 100_000
    population_size = 500
    dimension = 10

    params = LSHADE44Data(
        max_nfe=num_of_nfe,
        population_size=population_size,
        dimension=dimension,
        lb=[-100] * dimension,
        ub=[100] * dimension,

        show_plots=False,

        optimization_type=optimization.OptimizationType.MINIMIZATION,

        boundary_constraints_fun=boundary_constrain.BoundaryFixing.RANDOM,

        function=fitness_fun_opf,

        log_population=False,

        parallel_processing=["thread", 1],

        memory_size=5,
        best_member_percentage=0.2,
        smoothing_constant=2,
        reset_threshold=0.05,
        minimum_population_size=5,
        archive_size=100,
        pm_to_cr_table_size=100,
    )

    print("LSHADE44 parameters created successfully.")
    print("Starting LSHADE44...")

    start_time = time.time()

    monitor = TensorBoardMonitor(
        log_dir="runs",
        experiment_name="LSHADE44-11",
        every_epochs=10,
        every_nfe=10000,
        flush_every=1,
    )
    monitor_csv = CSVMonitor(log_dir="runs13", experiment_name="my_experiment", every_epochs=10, every_nfe=1000,
                             flush_every=5, )

    alg = LSHADE44(
        params,
        db_conn="LSHADE44.db",
        db_auto_write=False,
        db_writing_interval=1000,
        verbose=True,
        monitor=monitor_csv
    )

    results = alg.run()

    elapsed_time = time.time() - start_time

    best_fitness = min(
        epoch.best_individual.fitness_value
        for epoch in results.epoch_metrics
    )

    print()

    print("=" * 50)
    print("LSHADE44 TEST FINISHED")
    print("=" * 50)

    print(f"Best Fitness Value: {best_fitness}")
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")

    print()
    print("Strategy statistics:")
    alg.print_strategy_statistics()

    results.save_metrics_to_csv(
        output_dir="lshade44"
    )

    results.save_final_result_to_csv(
        output_dir="lshade44",
        filename="lshade44_result.csv"
    )
