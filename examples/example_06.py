import matplotlib.pyplot as plt
import time
import pandas as pd
from opfunu.cec_based import F12014

from detpy.DETAlgs.aade import AADE
from detpy.DETAlgs.al_shade import ALSHADE
from detpy.DETAlgs.comde import COMDE
from detpy.DETAlgs.data.alg_data import COMDEData, DEData, SADEData, FiADEData, ImprovedDEData, AADEData, DEGLData, \
    DELBData, DERLData, EIDEData, EMDEData, IDEData, JADEData, MGDEData, NMDEData, OppBasedData, \
    LSHADERSPData, ShadeData, LShadeData, ALSHADEData, SPSLShadeEIGDATA, LShadeEpsinData, Shade_1_1_Data
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
from detpy.DETAlgs.lshade import LSHADE
from detpy.DETAlgs.lshade_epsin import LShadeEpsin
from detpy.DETAlgs.lshadersp import LSHADERSP
from detpy.DETAlgs.mgde import MGDE
from detpy.DETAlgs.nmde import NMDE
from detpy.DETAlgs.opposition_based import OppBasedDE
from detpy.DETAlgs.sade import SADE
from detpy.DETAlgs.shade import SHADE
from detpy.DETAlgs.shade_1_1 import SHADE_1_1
from detpy.DETAlgs.sps_lshade_eig import SPS_LSHADE_EIG

from detpy.models.fitness_function import FitnessFunctionOpfunu
from detpy.models.enums import optimization, boundary_constrain

pd.set_option('display.float_format', lambda x: '%.6f' % x)


def extract_best_fitness(epoch_metrics):
    return [epoch.best_individual.fitness_value for epoch in epoch_metrics]


def run_algorithm(algorithm_class, params, db_conn="Differential_evolution.db", db_auto_write=False):
    start_time = time.time()
    algorithm = algorithm_class(params, db_conn=db_conn, db_auto_write=db_auto_write)
    results = algorithm.run()
    elapsed_time = time.time() - start_time
    fitness_values = [epoch.best_individual.fitness_value for epoch in results.epoch_metrics]
    best_fitness_value = min(fitness_values)
    avg_fitness = sum(fitness_values) / len(fitness_values)
    return avg_fitness, elapsed_time, best_fitness_value


def run_multiple_times(algorithm_class, params, runs=10):
    all_fitness = []
    all_times = []
    all_best_fitness = []

    for _ in range(runs):
        fitness, elapsed_time, best_fitness_value = run_algorithm(algorithm_class, params)
        all_fitness.append(fitness)
        all_times.append(elapsed_time)
        all_best_fitness.append(best_fitness_value)

    avg_fitness = sum(all_fitness) / len(all_fitness)
    avg_time = sum(all_times) / len(all_times)
    avg_best_fitness = sum(all_best_fitness) / len(all_best_fitness)

    return avg_fitness, avg_time, avg_best_fitness


def plot_fitness_convergence(fitness_results, algorithm_names, num_of_epochs):
    epochs = range(1, num_of_epochs + 1)
    for fitness_values, name in zip(fitness_results, algorithm_names):
        fitness_values = fitness_values[:num_of_epochs]
        plt.plot(epochs, fitness_values, label=name)

    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Best Fitness Value')
    plt.title('Fitness Convergence Algorithms')
    plt.subplots_adjust(right=0.8)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderpad=1.5, handlelength=2)
    plt.show()


if __name__ == "__main__":
    num_of_epochs = 100
    num_of_nfe = 10_000

    fitness_fun_opf = FitnessFunctionOpfunu(
        func_type=F12014,
        ndim=10
    )

    params_common = {
        'max_nfe': num_of_nfe,
        'population_size': 100,
        'dimension': 10,
        'lb': [-100] * 10,
        'ub': [100] * 10,
        'show_plots': False,
        'optimization_type': optimization.OptimizationType.MINIMIZATION,
        'boundary_constraints_fun': boundary_constrain.BoundaryFixing.RANDOM,
        'function': fitness_fun_opf,
        'log_population': True,
        'parallel_processing': ['thread', 1]
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
    params_shade = ShadeData(**params_common)
    params_shade11 = Shade_1_1_Data(**params_common)
    params_lshade = LShadeData(**params_common)
    params_lshadersp = LSHADERSPData(**params_common)
    params_lshadeepsin = LShadeEpsinData(**params_common)
    params_alshade = ALSHADEData(**params_common)

    algorithms = {
        "SADE": SADE,
        "COMDE": COMDE,
        'DERL': DERL,
        'NMDE': NMDE,
        "ALSHADE": ALSHADE,
        'SHADE1.1': SHADE_1_1,
        'SHADE': SHADE,
        'LSHADEEPSIN': LShadeEpsin,
        'LSHADE': LSHADE,
        'AADE': AADE,
        'DE': DE,
        'DEGL': DEGL,
        'DELB': DELB,
        'EIDE': EIDE,
        'EMDE': EMDE,
        'JADE': JADE,
        'MGDE': MGDE,
        'OppBased': OppBasedDE,
        'FiADE': FiADE,
        'IDE': IDE,
        'ImprovedDE': ImprovedDE,
        'LSHADERSP': LSHADERSP,
        'SPS_LSHADE_EIG': SPS_LSHADE_EIG,
    }

    algorithms_params = {
        'NMDE': params_nmde,
        'DERL': params_derl,
        'COMDE': params_comde,
        'LSHADE': params_lshade,
        'AADE': params_aade,
        'SADE': params_sade,
        'DE': params_de,
        'DEGL': params_degl,
        'DELB': params_delb,
        'EIDE': params_eide,
        'EMDE': params_emde,
        'FiADE': params_fiade,
        'IDE': params_ide,
        'ImprovedDE': params_improved_de,
        'JADE': params_jade,
        'MGDE': params_mgde,
        'OppBased': params_opposition_based,
        'LSHADEEPSIN': params_lshadeepsin,
        'SHADE': params_shade,
        'LSHADERSP': params_lshadersp,
        'ALSHADE': params_alshade,
        'SPS_LSHADE_EIG': SPSLShadeEIGDATA(**params_common),
        'SHADE1.1': params_shade11
    }

    avg_fitness_results = []
    avg_times = []
    algorithm_names = []
    avg_best_fitness_results = []

    for name, algorithm_class in algorithms.items():
        avg_fitness, avg_time, avg_best_fitness = run_multiple_times(algorithm_class, algorithms_params[name])
        avg_fitness_results.append(avg_fitness)
        avg_times.append(avg_time)
        algorithm_names.append(name)
        avg_best_fitness_results.append(avg_best_fitness)

    sorted_results = sorted(zip(algorithm_names, avg_fitness_results, avg_times, avg_best_fitness_results),
                            key=lambda x: (x[1], x[2], x[3]))

    df_sorted = pd.DataFrame(sorted_results,
                             columns=["Algorithm", "Average Fitness", "Average Time (s)", "Average Best Fitness"])

    df_fitness_sorted = df_sorted[["Algorithm", "Average Fitness"]].sort_values(by="Average Fitness")
    df_time_sorted = df_sorted[["Algorithm", "Average Time (s)"]].sort_values(by="Average Time (s)")
    df_best_fitness_sorted = df_sorted[["Algorithm", "Average Best Fitness"]].sort_values(by="Average Best Fitness")

    print("\nAverage Fitness (sorted by Fitness):")
    print(df_fitness_sorted)

    print("\nAverage Time (sorted by Time):")
    print(df_time_sorted)

    print("\nAverage Best Fitness (sorted by Best Fitness):")
    print(df_best_fitness_sorted)
