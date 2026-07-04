import matplotlib.pyplot as plt
from detpy.DETAlgs.data.alg_data import EPSDEwDCData, EPSRDEData, EPSDEGData, EPSADEData, EPSDEData, EPSDEAGData
from detpy.DETAlgs.eps_ade import EPSADE
from detpy.DETAlgs.eps_de import EPSDE
from detpy.DETAlgs.eps_de_w_dc import EPSDEwDC
from detpy.DETAlgs.eps_deag import EPSDEAG
from detpy.DETAlgs.eps_deg import EPSDEG
from detpy.DETAlgs.eps_rde import EPSRDE
from detpy.functions import FunctionLoader
from detpy.functions.function_loader import G01, G02, G03, G04, G05, G06, G07, G08, G09, G10, G11, G12
from detpy.models.fitness_function import BenchmarkFitnessFunction
from detpy.models.enums import boundary_constrain


def run_algorithm(algorithm_class, params, db_conn="Differential_evolution.db", db_auto_write=False):
    algorithm = algorithm_class(params, db_conn=db_conn, db_auto_write=db_auto_write)
    results = algorithm.run()
    return [epoch.best_individual.fitness_value for epoch in results.epoch_metrics]


def plot_fitness_convergence(fitness_results, algorithm_names, max_nfe, population_size, benchmark_name):
    max_epochs = int(max_nfe / population_size)
    for fitness_values, name in zip(fitness_results, algorithm_names):
        n = min(len(fitness_values), max_epochs)

        epochs = range(1, n + 1)
        y = fitness_values[:n]

        plt.plot(epochs, y, label=name)

    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Best Fitness Value')
    plt.title(f'Fitness Convergence of Algorithms on the {benchmark_name} Benchmark')
    plt.subplots_adjust(right=0.8)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderpad=1.5, handlelength=2)
    plt.show()


if __name__ == "__main__":
    num_of_nfe = 1000
    function_loader = FunctionLoader()
    algorithms = {
        'EPSDEDC': EPSDEwDC,
        'EPSRDE': EPSRDE,
        'EPSDE': EPSDE,
        'EPSDEG': EPSDEG,
        'EPSDEAG': EPSDEAG,
        'EPSADE': EPSADE
    }

    benchmarks = [G01(13), G02(20), G03(10), G04(5), G05(4), G06(2), G07(10), G08(2), G09(7), G10(8), G11(2), G12(3)]

    for benchmark in benchmarks:
        param = {
            'max_nfe': num_of_nfe,
            'population_size': 100,
            'dimension': benchmark.n_dimensions,
            'lb': benchmark.lower_bounds,
            'ub': benchmark.upper_bounds,
            'optimization_type': benchmark.optimization_type,
            'boundary_constraints_fun': boundary_constrain.BoundaryFixing.RANDOM,
            'function': BenchmarkFitnessFunction(
                function_loader.get_function(function_name=benchmark.name, n_dimensions=benchmark.n_dimensions)),
            'g_funcs': benchmark.g,
            'h_funcs': benchmark.h,
            'log_population': True,
            'parallel_processing': ['thread', 5]
        }

        algorithms_params = {
            'EPSDEDC': EPSDEwDCData(**param),
            'EPSRDE': EPSRDEData(**param),
            'EPSDE': EPSDEData(**param),
            'EPSDEG': EPSDEGData(**param),
            'EPSDEAG': EPSDEAGData(**param),
            'EPSADE': EPSADEData(**param)
        }

        fitness_results = []
        algorithm_names = []
        for name, algorithm_class in algorithms.items():
            algorithm_names.append(name)
            fitness_values = run_algorithm(algorithm_class, algorithms_params[name], db_auto_write=False)
            fitness_results.append(fitness_values)

        plot_fitness_convergence(fitness_results, algorithm_names, num_of_nfe, param["population_size"], benchmark.name)
