from typing import Sequence

import numpy as np

from detpy.models.enums.optimization import OptimizationType


def paper_initial_population_size(dimension: int, r_n_init: int) -> int:
    """Section III-A: N_init = round(D * r_Ninit)."""
    return int(round(dimension * r_n_init))


def archive_capacity(population_size: int, r_arc: float) -> int:
    """Section III-A: |A| = round(N * r_arc), resized with the current population."""
    return int(round(population_size * r_arc))


def repair_coordinate(value: float, parent: float, lower: float, upper: float) -> float:
    """Eq. (4): pull an infeasible mutant coordinate halfway back to its parent."""
    if value < lower:
        return (lower + parent) / 2.0
    if value > upper:
        return (upper + parent) / 2.0
    return value


def weighted_lehmer_mean(values: Sequence[float], deltas: Sequence[float]) -> float:
    """Eqs. (7)-(9): weighted Lehmer mean, weights proportional to fitness improvement."""
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
    """
    DPSR stage index in ``0 .. rfreq - 1``.

    Brest and Maučec give every population size the same share of the
    evaluation budget, ``maxnfeval / pmax``. ``rfreq`` is that ``pmax``.
    The number of completed halvings equals the stage index.
    """
    if reduction_frequency <= 1 or nfe_max <= 0:
        return 0
    stage = (int(nfe) * int(reduction_frequency)) // int(nfe_max)
    return min(int(reduction_frequency) - 1, stage)


def population_size_after_reductions(start_size: int, reductions: int, minimum_size: int) -> int:
    """Halve ``reductions`` times, and stop when the next half would fall below ``minimum_size``."""
    size = int(start_size)
    for _ in range(max(0, int(reductions))):
        half = size // 2
        if half < minimum_size:
            break
        size = half
    return size


def reduce_population_once(members: Sequence, optimization: OptimizationType) -> list:
    """
    One DPSR step from Brest and Maučec.

    Compare individual ``i`` with individual ``i + N/2`` and keep the better one.
    On a tie the first-half individual stays. An odd leftover individual is dropped.
    """
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
    """Apply the pair-halving step until the population reaches ``target_size``."""
    current = list(members)
    while len(current) > target_size:
        half = len(current) // 2
        if half < minimum_size:
            break
        current = reduce_population_once(current, optimization)
    return current
