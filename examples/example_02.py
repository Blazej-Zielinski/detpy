import matplotlib.pyplot as plt
from detpy.DETAlgs.aade import AADE
from detpy.DETAlgs.comde import COMDE
from detpy.DETAlgs.data.alg_data import COMDEData, DEData, SADEData, FiADEData, ImprovedDEData, AADEData, DEGLData, \
    DELBData, DERLData, EIDEData, EMDEData, IDEData, JADEData, MGDEData, NMDEData, OppBasedData
from detpy.DETAlgs.de import DE
from detpy.DETAlgs.degl import DEGL
from detpy.DETAlgs.delb import DELB
from detpy.DETAlgs.derl import DERL
from detpy.DETAlgs.eide import EIDE
from detpy.DETAlgs.emde import EMDE
from detpy.DETAlgs.fiade import FiADE
from detpy.DETAlgs.ide import IDE
from detpy.DETAlgs.improved_de import ImprovedDE
from detpy.DETAlgs.jade import JADE
from detpy.DETAlgs.mgde import MGDE
from detpy.DETAlgs.nmde import NMDE
from detpy.DETAlgs.opposition_based import OppBasedDE
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


def plot_fitness_convergence(fitness_results, algorithm_names, max_nfe, population_size):
    max_epochs = int(max_nfe / population_size)
    for fitness_values, name in zip(fitness_results, algorithm_names):
        n = min(len(fitness_values), max_epochs)

        epochs = range(1, n + 1)
        y = fitness_values[:n]

        plt.plot(epochs, y, label=name)

    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Best Fitness Value')
    plt.title('Fitness Convergence Algorithms')
    plt.subplots_adjust(right=0.8)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderpad=1.5, handlelength=2)
    plt.show()


if __name__ == "__main__":
    num_of_nfe = 1000
    function_loader = FunctionLoader()
    ackley_function = function_loader.get_function(function_name="ackley", n_dimensions=2)
    fitness_fun = BenchmarkFitnessFunction(ackley_function)

    params_common = {
        'max_nfe': num_of_nfe,
        'population_size': 50,
        'dimension': 2,
        'lb': [-32.768, -32.768],
        'ub': [32.768, 32.768],
        'optimization_type': optimization.OptimizationType.MINIMIZATION,
        'boundary_constraints_fun': boundary_constrain.BoundaryFixing.RANDOM,
        'function': fitness_fun,
        'log_population': True,
        'parallel_processing': ['thread', 5]
    }
    params_aade = AADEData(**params_common)
    params_comde = COMDEData(**params_common)
    params_de = DEData(**params_common)
    params_degl = DEGLData(**params_common)
    params_delb = DELBData(**params_common)
    params_derl = DERLData(**params_common)
    params_eide = EIDEData(**params_common)
    params_emde = EMDEData(**params_common)
    params_fiade = FiADEData(**params_common)
    params_ide = IDEData(**params_common)
    params_improved_de = ImprovedDEData(**params_common)
    params_jade = JADEData(**params_common)
    params_mgde = MGDEData(**params_common)
    params_nmde = NMDEData(**params_common)
    params_opposition_based = OppBasedData(**params_common)
    params_sade = SADEData(**params_common)
    #
    fitness_aade = run_algorithm(AADE, params_aade)
    fitness_comde = run_algorithm(COMDE, params_comde)
    fitness_de = run_algorithm(DE, params_de)
    fitness_degl = run_algorithm(DEGL, params_degl)
    fitness_delb = run_algorithm(DELB, params_delb)
    fitness_derl = run_algorithm(DERL, params_derl)
    fitness_eide = run_algorithm(EIDE, params_eide)
    fitness_emde = run_algorithm(EMDE, params_emde)
    fitness_fiade = run_algorithm(FiADE, params_fiade)
    fitness_ide = run_algorithm(IDE, params_ide)
    fitness_improved_de = run_algorithm(ImprovedDE, params_improved_de)
    fitness_jade = run_algorithm(JADE, params_jade)
    fitness_mgde = run_algorithm(MGDE, params_mgde)
    fitness_ndme = run_algorithm(NMDE, params_nmde)
    fitness_opposition_based = run_algorithm(OppBasedDE, params_opposition_based)
    fitness_sade = run_algorithm(SADE, params_sade)

    fitness_results = [fitness_aade, fitness_comde, fitness_de, fitness_degl, fitness_delb, fitness_derl, fitness_eide,
                       fitness_emde, fitness_fiade,
                       fitness_ide, fitness_improved_de, fitness_jade, fitness_mgde, fitness_ndme,
                       fitness_opposition_based, fitness_sade]

    algorithm_names = ['AADE', 'COMDE', 'DE', 'DEGL', 'DELB', 'DERL', 'EIDE', 'EMDE', 'FiADE', 'IDE', 'ImprovedDE',
                       'JADE', 'MGDE', 'NMDE', 'OppBasedDE', 'SADE']

    plot_fitness_convergence(fitness_results, algorithm_names, num_of_nfe, params_common["population_size"])
