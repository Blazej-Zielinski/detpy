from typing import Sequence

import numpy as np

from detpy.models.enums.optimization import OptimizationType


def paper_initial_population_size(dimension: int, r_n_init: int) -> int:
    return int(round(dimension * r_n_init))


def archive_capacity(population_size: int, r_arc: float) -> int:
    return int(round(population_size * r_arc))


def repair_coordinate(value: float, parent: float, lower: float, upper: float) -> float:
    if value < lower:
        return (lower + parent) / 2.0
    if value > upper:
        return (upper + parent) / 2.0
    return value


def weighted_lehmer_mean(values: Sequence[float], deltas: Sequence[float]) -> float:
    samples = np.asarray(values, dtype=float)
    improvements = np.asarray(deltas, dtype=float)
    total = float(np.sum(improvements))
    if total <= 0.0 or samples.size == 0:
        return 0.0
    weights = improvements / total
    denominator = float(np.sum(weights * samples))
    if denominator == 0.0:
        return 0.0
    return float(np.sum(weights * samples * samples) / denominator)


def dpsr_stage(nfe: int, nfe_max: int, reduction_frequency: int) -> int:
    if reduction_frequency <= 1 or nfe_max <= 0:
        return 0
    stage = (int(nfe) * int(reduction_frequency)) // int(nfe_max)
    return min(int(reduction_frequency) - 1, stage)


def population_size_after_reductions(start_size: int, reductions: int, minimum_size: int) -> int:
    size = int(start_size)
    for _ in range(max(0, int(reductions))):
        half = size // 2
        if half < minimum_size:
            break
        size = half
    return size


def reduce_population_once(members: Sequence, optimization: OptimizationType) -> list:
    current = list(members)
    half = len(current) // 2
    if half < 1:
        return current

    kept = []
    for i in range(half):
        left = current[i]
        right = current[half + i]
        if optimization == OptimizationType.MINIMIZATION:
            better = left if left.fitness_value <= right.fitness_value else right
        else:
            better = left if left.fitness_value >= right.fitness_value else right
        kept.append(better)
    return kept


def reduce_population_to(members: Sequence, target_size: int, minimum_size: int,
                         optimization: OptimizationType) -> list:
    current = list(members)
    while len(current) > target_size:
        half = len(current) // 2
        if half < minimum_size:
            break
        current = reduce_population_once(current, optimization)
    return current
