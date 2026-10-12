import math
from typing import List, Sequence

import numpy as np

from detpy.models.enums.optimization import OptimizationType


def estimate_generation_max(nfe_max: int, start_pop_size: int, min_pop_size: int) -> int:
    """
    Number of generations implied by Eq. (3) until FEs reaches FEs_max.
    Initialization already spends start_pop_size evaluations.
    """
    nfe = start_pop_size
    size = start_pop_size
    generations = 0
    while nfe < nfe_max:
        nfe += size
        generations += 1
        size = int(round((min_pop_size - start_pop_size) * min(nfe, nfe_max) / nfe_max + start_pop_size))
        size = max(min_pop_size, size)
    return max(1, generations)


def sinusoidal_decreasing(freq: float, generation: int, generation_max: int) -> float:
    """
    Equation (4):
    F = 1/2 * (sin(2π * freq * g + π) * (Gmax - g)/Gmax + 1)
    """
    remaining = (generation_max - generation) / generation_max
    return 0.5 * (math.sin(2.0 * math.pi * freq * generation + math.pi) * remaining + 1.0)


def sinusoidal_increasing(freq: float, generation: int, generation_max: int) -> float:
    """
    Equation (5):
    F = 1/2 * (sin(2π * freq_i * g) * (g/Gmax) + 1)
    """
    progress = generation / generation_max
    return 0.5 * (math.sin(2.0 * math.pi * freq * generation) * progress + 1.0)


def configuration_success_rate(successes: Sequence[int], failures: Sequence[int],
                               epsilon: float = 0.01) -> float:
    """
    Equation (8):
    S_k,g = (sum ns) / (sum ns + sum nf) + ε , ε = 0.01
    """
    ns = float(sum(successes))
    nf = float(sum(failures))
    if ns + nf == 0.0:
        return epsilon
    return ns / (ns + nf) + epsilon


def configuration_probabilities(success_rates: Sequence[float]) -> List[float]:
    """Equation (7): p_k = S_k / sum_j S_j."""
    total = float(sum(success_rates))
    if total <= 0.0:
        share = 1.0 / len(success_rates)
        return [share] * len(success_rates)
    return [rate / total for rate in success_rates]


def choose_larger_probability_config(probabilities: Sequence[float]) -> int:
    """Fig. 1 line 12: use the sinusoidal configuration with larger probability."""
    if probabilities[0] > probabilities[1]:
        return 0
    if probabilities[1] > probabilities[0]:
        return 1
    return 0 if np.random.rand() < 0.5 else 1


def weighted_lehmer_mean(values: Sequence[float], deltas: Sequence[float]) -> float:
    """Equations (13)-(15): mean_WL(S) with w_k = Δf_k / sum(Δf)."""
    values = np.asarray(values, dtype=float)
    deltas = np.asarray(deltas, dtype=float)
    total = np.sum(deltas)
    if total <= 0.0:
        return 0.0
    weights = deltas / total
    denominator = np.sum(weights * values)
    if denominator == 0.0:
        return 0.0
    return float(np.sum(weights * values * values) / denominator)


def current_to_pbest_values(x: np.ndarray, pbest: np.ndarray, r1: np.ndarray, r2: np.ndarray,
                            f: float) -> np.ndarray:
    """Equation (2): V = X + F (X_pbest - X) + F (X_r1 - X_r2)."""
    return x + f * (pbest - x) + f * (r1 - r2)


def member_values(member) -> np.ndarray:
    return np.array([chromosome.real_value for chromosome in member.chromosomes], dtype=float)


def set_member_values(member, values: np.ndarray):
    for i, value in enumerate(values):
        member.chromosomes[i].real_value = float(value)
    return member


def euclidean_neighborhood_indices(points: np.ndarray, best_index: int,
                                   fraction: float) -> np.ndarray:
    """NP × ps nearest individuals to X_best by Euclidean distance."""
    distances = np.linalg.norm(points - points[best_index], axis=1)
    neighborhood_size = max(2, int(len(points) * fraction))
    neighborhood_size = min(neighborhood_size, len(points))
    return np.argsort(distances)[:neighborhood_size]


def covariance_eigenvectors(neighborhood_points: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    """B from C = B D B^T (eq. 16)."""
    dimension = neighborhood_points.shape[1]
    if neighborhood_points.shape[0] < 2:
        return np.eye(dimension)

    covariance = np.cov(neighborhood_points, rowvar=False)
    if np.ndim(covariance) == 0:
        covariance = np.array([[float(covariance)]])
    covariance = covariance + epsilon * np.eye(covariance.shape[0])
    _, eigenvectors = np.linalg.eigh(covariance)
    return eigenvectors


def binomial_crossover_values(target: np.ndarray, mutant: np.ndarray, cr: float) -> np.ndarray:
    """Equation (19): u = v if rand <= CR or j = j_rand, else x."""
    mask = np.random.rand(len(target)) <= cr
    mask[np.random.randint(0, len(target))] = True
    return np.where(mask, mutant, target)


def eigen_binomial_crossover(target: np.ndarray, mutant: np.ndarray, cr: float,
                             eigenvectors: np.ndarray) -> np.ndarray:
    """Equations (17)-(20): X' = B^T X, V' = B^T V, binomial, U = B U'."""
    target_eigen = eigenvectors.T @ target
    mutant_eigen = eigenvectors.T @ mutant
    trial_eigen = binomial_crossover_values(target_eigen, mutant_eigen, cr)
    return eigenvectors @ trial_eigen


def is_better(origin_fitness: float, trial_fitness: float, optimization: OptimizationType) -> bool:
    if optimization == OptimizationType.MINIMIZATION:
        return trial_fitness < origin_fitness
    return trial_fitness > origin_fitness
