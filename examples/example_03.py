import matplotlib.pyplot as plt
from detpy.DETAlgs.data.alg_data import SADEData
from detpy.DETAlgs.sade import SADE
from detpy.functions import FunctionLoader
from detpy.models.fitness_function import BenchmarkFitnessFunction
from detpy.models.enums import optimization, boundary_constrain


def extract_best_fitness(epoch_metrics):
    return [epoch.best_individual.fitness_value for epoch in epoch_metrics]


def run_algorithm(algorithm_class, params, db_conn="Differential_evolution.db", db_auto_write=False):
    algorithm = algorithm_class(params, db_conn=db_conn, db_auto_write=db_auto_write)
    results = algorithm.run()
    return [epoch.best_individual.fitness_value for epoch in results.epoch_metrics]


def plot_fitness_convergence(fitness_results, algorithm_names, num_of_epochs, function_name):
    epochs = range(1, num_of_epochs + 1)
    for fitness_values, name in zip(fitness_results, algorithm_names):
        fitness_values = fitness_values[:num_of_epochs]
        plt.plot(epochs, fitness_values, label=name)

    plt.xlabel('Epoch')
    plt.ylabel('Best Fitness Value')
    plt.title(f'Fitness Convergence for {function_name}')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    num_of_epochs = 100
    function_loader = FunctionLoader()

    test_functions = [
        "ackley", "rastrigin", "rosenbrock"
    ]

    params_common = {
        'epoch': num_of_epochs,
        'population_size': 100,
        'dimension': 2,
        'lb': [-32.768, -32.768],
        'ub': [32.768, 32.768],
        'optimization_type': optimization.OptimizationType.MINIMIZATION,
        'boundary_constraints_fun': boundary_constrain.BoundaryFixing.RANDOM,
        'log_population': True,
        'parallel_processing': ['thread', 5]
    }

    algorithms = [
        (SADE, SADEData),
    ]
    algorithm_names = ['SADE']

    for function_name in test_functions:
        fitness_fun = BenchmarkFitnessFunction(
            function_loader.get_function(function_name=function_name, n_dimensions=2))
        params_common['function'] = fitness_fun

        fitness_results = []
        for algorithm_class, data_class in algorithms:
            params = data_class(**params_common)
            fitness_values = run_algorithm(algorithm_class, params)
            fitness_results.append(fitness_values)

        plot_fitness_convergence(fitness_results, algorithm_names, num_of_epochs, function_name)
