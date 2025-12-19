import matplotlib.pyplot as plt
import time
import pandas as pd
from detpy.DETAlgs.aade import AADE
from detpy.DETAlgs.al_shade import ALSHADE
from detpy.DETAlgs.data.alg_data import COMDEData, DEData, SADEData, FiADEData, ImprovedDEData, AADEData, DEGLData, \
    DELBData, DERLData, EIDEData, EMDEData, IDEData, JADEData, MGDEData, NMDEData, OppBasedData, \
    LSHADERSPData, ShadeData, LShadeData, ALSHADEData, SPSLShadeEIGDATA, LShadeEpsinData, Shade_1_1_Data
from detpy.DETAlgs.de import DE
from detpy.DETAlgs.degl import DEGL
from detpy.DETAlgs.delb import DELB
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
from detpy.DETAlgs.opposition_based import OppBasedDE
from detpy.DETAlgs.shade import SHADE
from detpy.DETAlgs.shade_1_1 import SHADE_1_1
from detpy.DETAlgs.sps_lshade_eig import SPS_LSHADE_EIG

from detpy.functions import FunctionLoader
from detpy.models.fitness_function import BenchmarkFitnessFunction
from detpy.models.enums import optimization, boundary_constrain
from detpy.models.stop_condition.never_stop_condition import NeverStopCondition

pd.set_option('display.float_format', lambda x: '%.6f' % x)

def run_algorithm(algorithm_class, params, target_value=100, db_conn="Differential_evolution.db", db_auto_write=False):
    start_time = time.time()  # Start measuring time
    algorithm = algorithm_class(params, db_conn=db_conn, db_auto_write=db_auto_write)
    results = algorithm.run()
    elapsed_time = time.time() - start_time  # Measure elapsed time

    # Pobierz wartości funkcji przystosowania
    fitness_values = [epoch.best_individual.fitness_value for epoch in results.epoch_metrics]

    # Oblicz błąd względem docelowej wartości
    errors = [target_value - value for value in fitness_values]

    # Oblicz wskaźniki na podstawie błędów
    avg_error = sum(errors) / len(errors)
    min_error = min(errors)  # Najmniejszy błąd (najbliżej docelowej wartości)

    return avg_error, elapsed_time, min_error


def run_multiple_times(algorithm_class, params, target_value=10, runs=10):
    all_errors = []
    all_times = []
    all_min_errors = []

    for _ in range(runs):
        avg_error, elapsed_time, min_error = run_algorithm(algorithm_class, params, target_value)
        all_errors.append(avg_error)
        all_times.append(elapsed_time)
        all_min_errors.append(min_error)

    # Calculate average error and average time
    avg_error = sum(all_errors) / len(all_errors)
    avg_time = sum(all_times) / len(all_times)
    avg_min_error = sum(all_min_errors) / len(all_min_errors)

    return avg_error, avg_time, avg_min_error


if __name__ == "__main__":
    num_of_epochs = 100
    num_of_nfe = 1500
    function_loader = FunctionLoader()
    ackley_function = function_loader.get_function(function_name="cec2013_f1_opfunu_max", n_dimensions=10)
    fitness_fun = BenchmarkFitnessFunction(ackley_function)

    params_common = {
        'max_nfe': num_of_nfe,
        'population_size': 100,
        'dimension': 10,
        'lb': [-100]*10,
        'ub': [100]*10,
        'show_plots': False,
        'optimization_type': optimization.OptimizationType.MAXIMIZATION,
        'boundary_constraints_fun': boundary_constrain.BoundaryFixing.RANDOM,
        'function': fitness_fun,
        'log_population': True,
        'parallel_processing': ['thread', 1]
    }

    params_alshade = {
        'max_nfe': num_of_nfe,
        'population_size': 100,
        'dimension': 10,
        'show_plots': False,
        'lb': [-100] * 10,
        'ub': [100] * 10,
        'additional_stop_criteria': NeverStopCondition(),
        'optimization_type': optimization.OptimizationType.MAXIMIZATION,
        'function': fitness_fun,
        'log_population': True,
        'parallel_processing': ['thread', 1],
        'boundary_constraints_fun': boundary_constrain.BoundaryFixing.WITH_PARENT,
    }

    params_lshade = LShadeData(**params_alshade)
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
    params_lshadersp = LSHADERSPData(**params_alshade)
    params_lshadeepsin = LShadeEpsinData(**params_alshade)
    params_shade11 = Shade_1_1_Data(**params_alshade)
    params_als_hade = ALSHADEData(**params_alshade)

    algorithms = {
        "ALSHADE": ALSHADE,
        # "SHADE1.1": SHADE_1_1,
        # "SHADE": SHADE,
        # "LSHADEEPSIN": LShadeEpsin,
        # "LSHADE": LSHADE,
        # "LSHADERSP": LSHADERSP,
        # "SPS_LSHADE_EIG": SPS_LSHADE_EIG,
    }

    algorithms_params = {
        'LSHADE': params_lshade,
        'AADE': params_aade,
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
        'ALSHADE': params_als_hade,
        'SPS_LSHADE_EIG': SPSLShadeEIGDATA(**params_common),
        'SHADE1.1': params_shade11
    }

    avg_error_results = []
    avg_times = []
    algorithm_names = []
    avg_min_error_results = []

    for name, algorithm_class in algorithms.items():
        avg_error, avg_time, avg_min_error = run_multiple_times(algorithm_class, algorithms_params[name])
        avg_error_results.append(avg_error)
        avg_times.append(avg_time)
        algorithm_names.append(name)
        avg_min_error_results.append(avg_min_error)

    sorted_results = sorted(zip(algorithm_names, avg_error_results, avg_times, avg_min_error_results),
                            key=lambda x: (x[1], x[2], x[3]))

    df_sorted = pd.DataFrame(sorted_results, columns=["Algorithm", "Average Error", "Average Time (s)", "Average Min Error"])

    print("\nAverage Error (sorted by Error):")
    print(df_sorted[["Algorithm", "Average Error"]].sort_values(by="Average Error"))

    print("\nAverage Time (sorted by Time):")
    print(df_sorted[["Algorithm", "Average Time (s)"]].sort_values(by="Average Time (s)"))

    print("\nAverage Min Error (sorted by Min Error):")
    print(df_sorted[["Algorithm", "Average Min Error"]].sort_values(by="Average Min Error"))