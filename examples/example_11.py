
import os
import csv
import time

from opfunu.cec_based import F12014

from detpy.DETAlgs.lshade import LSHADE
from detpy.DETAlgs.data.alg_data import LShadeData
from detpy.models.fitness_function import FitnessFunctionOpfunu
from detpy.models.enums import optimization, boundary_constrain


def get_available_cpus():
    """Return the number of logical CPUs available to the current process."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def benchmark_lshade():
    # Define optimization parameters
    num_of_nfe = 100_000
    population_size = 100
    dimension = 10

    # Detect the maximum number of available logical CPUs
    max_cpus = get_available_cpus()

    # Generate worker counts: 2, 4, 6, ... up to the platform maximum
    worker_counts = list(range(2, max_cpus + 1, 2))

    # Include the maximum CPU count if it is not already in the list
    if max_cpus >= 2 and max_cpus not in worker_counts:
        worker_counts.append(max_cpus)

    # Define execution modes: sequential, process-based, and thread-based
    configurations = [("none", None)]

    for workers in worker_counts:
        configurations.append(("process", ["process", workers]))
        configurations.append(("thread", ["thread", workers]))

    print(f"Available logical CPUs: {max_cpus}")
    print(f"Number of configurations: {len(configurations)}")

    results_table = []

    for mode, parallel_config in configurations:
        print(
            f"\nTesting mode: {mode}, "
            f"configuration: {parallel_config}"
        )

        # Initialize the fitness function for the current run
        fitness_fun = FitnessFunctionOpfunu(
            func_type=F12014,
            ndim=dimension
        )

        # Configure the LSHADE algorithm
        params = LShadeData(
            max_nfe=num_of_nfe,
            population_size=population_size,
            dimension=dimension,
            lb=[-100] * dimension,
            ub=[100] * dimension,
            show_plots=False,
            optimization_type=optimization.OptimizationType.MINIMIZATION,
            boundary_constraints_fun=boundary_constrain.BoundaryFixing.RANDOM,
            function=fitness_fun,
            log_population=False,
            parallel_processing=parallel_config,
            archive_ratio=1.1
        )

        # Disable monitoring to minimize measurement overhead
        lshade = LSHADE(
            params,
            db_conn=None,
            db_auto_write=False,
            verbose=False,
            monitor=None
        )

        # Measure the total execution time
        start = time.perf_counter()
        results = lshade.run()
        elapsed = time.perf_counter() - start

        # Extract the best fitness value obtained during optimization
        best_fitness = min(
            epoch.best_individual.fitness_value
            for epoch in results.epoch_metrics
        )

        # Represent sequential execution as a single worker
        workers = (
            1 if parallel_config is None
            else parallel_config[1]
        )

        # Store the benchmark results
        row = {
            "mode": mode,
            "workers": workers,
            "elapsed_seconds": elapsed,
            "best_fitness": best_fitness
        }
        results_table.append(row)

        print(
            f"Execution time: {elapsed:.2f} s | "
            f"Best fitness: {best_fitness:.8g}"
        )

    # Export benchmark results to a CSV file
    with open(
        "lshade_parallel_benchmark.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "mode",
                "workers",
                "elapsed_seconds",
                "best_fitness"
            ]
        )
        writer.writeheader()
        writer.writerows(results_table)

    # Sort configurations by execution time
    print("\n=== Benchmark ranking by execution time ===")

    for row in sorted(
        results_table,
        key=lambda r: r["elapsed_seconds"]
    ):
        print(
            f"{row['mode']:8} | "
            f"workers={row['workers']:2} | "
            f"time={row['elapsed_seconds']:.2f} s | "
            f"fitness={row['best_fitness']:.8g}"
        )

    print("\nResults saved to lshade_parallel_benchmark.csv")


if __name__ == "__main__":
    benchmark_lshade()
