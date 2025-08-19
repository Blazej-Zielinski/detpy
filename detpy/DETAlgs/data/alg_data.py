from dataclasses import dataclass, field
from typing import Optional, Callable

from detpy.models.enums.basevectorschema import BaseVectorSchema
from detpy.models.enums.crossingtype import CrossingType
from detpy.models.enums.derivative_method import DerivativeMethod
from detpy.models.fitness_function import FitnessFunctionBase
from detpy.models.enums.boundary_constrain import BoundaryFixing
from detpy.models.enums.optimization import OptimizationType


@dataclass
class BaseData:
    epoch: int = 100
    population_size: int = 100
    dimension: int = 10
    lb: list = field(default_factory=lambda: [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100])
    ub: list = field(default_factory=lambda: [100, 100, 100, 100, 100, 100, 100, 100, 100, 100])
    optimization_type: OptimizationType = OptimizationType.MINIMIZATION
    boundary_constraints_fun: BoundaryFixing = BoundaryFixing.RANDOM
    function: FitnessFunctionBase = None
    log_population: bool = False
    parallel_processing: Optional[list] = None


@dataclass
class DEData(BaseData):
    mutation_factor: float = 0.5
    crossover_rate: float = 0.5
    crossing_type: CrossingType = CrossingType.BINOMIAL
    y: int = 1
    base_vector_schema: BaseVectorSchema = BaseVectorSchema.CURRENT


@dataclass
class COMDEData(BaseData):
    mutation_factor: float = 0.1
    crossover_rate: float = 0.1
    crossing_type: CrossingType = CrossingType.BINOMIAL


@dataclass
class DERLData(BaseData):
    mutation_factor: float = 0.1
    crossover_rate: float = 0.1
    crossing_type: CrossingType = CrossingType.BINOMIAL


@dataclass
class NMDEData(BaseData):
    delta_f: float = 0.1
    delta_cr: float = 0.1
    sp: int = 10


@dataclass
class SADEData(BaseData):
    prob_f: float = 0.1
    prob_cr: float = 0.1


@dataclass
class EMDEData(BaseData):
    crossover_rate: float = 0.1
    crossing_type: CrossingType = CrossingType.BINOMIAL


@dataclass
class IDEData(BaseData):
    base_vector_schema: BaseVectorSchema = BaseVectorSchema.CURRENT
    y: int = 1


@dataclass
class DELBData(BaseData):
    crossover_rate: float = 0.1
    w_factor: float = 0.1  # control frequency of local exploration around trial and best vectors
    crossing_type: CrossingType = CrossingType.BINOMIAL


@dataclass
class OppBasedData(BaseData):
    mutation_factor: float = 0.1
    crossover_rate: float = 0.1
    crossing_type: CrossingType = CrossingType.BINOMIAL
    y: int = 1
    base_vector_schema: BaseVectorSchema = BaseVectorSchema.CURRENT
    max_nfc: float = 0.1
    jumping_rate: float = 0.1


@dataclass
class DEGLData(BaseData):
    mutation_factor: float = 0.1
    crossover_rate: float = 0.1
    crossing_type: CrossingType = CrossingType.BINOMIAL
    radius: int = 10  # neighborhood size, 2k + 1 <= NP, at least k=2
    weight: float = 0.1  # controls the balance between the exploration and exploitation


@dataclass
class JADEData(BaseData):
    archive_size: int = 10
    mutation_factor_mean: float = 0.1
    mutation_factor_std: float = 0.1
    crossover_rate_mean: float = 0.1
    crossover_rate_std: float = 0.1
    crossover_rate_low: float = 0.1
    crossover_rate_high: float = 0.1
    c: float = 0.1  # describes the rate of parameter adaptation
    p: float = 0.1  # describes the greediness of the mutation strategy


@dataclass
class AADEData(BaseData):
    mutation_factor: float = 0.1
    crossover_rate: float = 0.1


@dataclass
class EIDEData(BaseData):
    crossover_rate_min: float = 0.1
    crossover_rate_max: float = 0.1
    crossing_type: CrossingType = CrossingType.BINOMIAL
    y: int = 1
    base_vector_schema: BaseVectorSchema = BaseVectorSchema.CURRENT


@dataclass
class MGDEData(BaseData):
    crossover_rate: float = 0.1
    crossing_type: CrossingType = CrossingType.BINOMIAL
    mutation_factor_f: float = 0.1
    mutation_factor_k: float = 0.1
    threshold: float = 0.1
    mu: float = 0.1


@dataclass
class FiADEData(BaseData):
    mutation_factor: float = 0.5
    crossover_rate: float = 0.5
    adaptive: bool = True


@dataclass
class ImprovedDEData(BaseData):
    mutation_factor: float = 0.1
    crossover_rate: float = 0.5
    crossing_type: CrossingType = CrossingType.BINOMIAL


@dataclass
class ShadeData(BaseData):
    memory_size: int = 5
    archive_size: int = 10


@dataclass
class LSHADERSPData(BaseData):
    scaling_factor_for_rank_selection: float = 3.5
    memory_size: int = 5
    minimum_population_size: int = 20


@dataclass
class LShadeData(BaseData):
    minimum_population_size: int = 5
    memory_size: int = 5


@dataclass
class SPSLShadeEIGDATA(BaseData):
    minimum_population_size: int = 5
    memory_size: int = 20
    q: int = 64
    f_init: float = 0.5
    cr_init: float = 0.3
    er_init: float = 1.0
    cr_min: float = 0.6
    cr_max: float = 0.95
    learning_rate_init: float = 0.1
    p_best_fraction: float = 0.1
    w_ext: float = 1.90
    w_er: float = 0.6807
    w_cr: float = 0.2079
    w_f: float = 0.3530


@dataclass
class ALSHADEData(BaseData):
    minimum_population_size: int = 4
    memory_size: int = 6
    elite_factor: float = 0.5
    init_probability_mutation_strategy: float = 0.5

@dataclass
class DEACRData(BaseData):
    triangular_distribution_for_crossover_rate = [0.2, 0.5, 1.0]
    triangular_distribution_for_mutation_factory = [0.3, 0.4, 0.5]
    number_of_success_crossover_rate: int = 15
    lineal_recombination_factor: float = 0.75
    gamma_var: int = 3

@dataclass
class LShadeEpsinData(BaseData):
    minimum_population_size: int = 5
    memory_size: int = 5
    elite_factor: float = 0.5
    init_probability_mutation_strategy: float = 0.5
    freg: float = 0.1

@dataclass
class EPSDEData(BaseData):
    mutation_factor : float = 0.7
    crossover_rate : float = 0.9
    epsilon_level : int = 0
    penalty_power : int = 2
    g_funcs: list[Callable[[list[float]], float]] = field(default_factory=list)
    h_funcs : list[Callable[[list[float]], float]] = field(default_factory=list)

@dataclass
class EPSDEAGData(BaseData):
    init_mutation_factor : float = 0.5
    init_crossover_rate : float = 0.9
    theta : float = 0.9
    penalty_power : int = 2
    control_generations : int = 150
    archive_size: int = 300
    gradient_base_mutation_rate : float = 0.2
    number_of_repeating_mutation = 3
    number_of_repeating_de_operations = 2
    gradient_mutation_interval = 5
    derivative_method = DerivativeMethod.NUMERIC
    g_funcs: list[Callable[[list[float]], float]] = field(default_factory=list)
    h_funcs : list[Callable[[list[float]], float]] = field(default_factory=list)