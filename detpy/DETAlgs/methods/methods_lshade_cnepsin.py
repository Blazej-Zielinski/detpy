import math

import numpy as np

from detpy.DETAlgs.crossover_methods.binomial_crossover import binomial_crossover_values


def round_half_up(value):
    """Round half away from zero for the positive population sizes in Eq. 3."""
    return int(math.floor(float(value) + 0.5))


def linear_population_size(nfe, nfe_max, start_pop_size, min_pop_size):
    """
    Population size for the next generation, Eq. 3.

    NP = Round((NPmin - NPmax) / maxFEs * FEs + NPmax)
    """
    raw_size = (min_pop_size - start_pop_size) * (nfe / nfe_max) + start_pop_size
    return round_half_up(raw_size)


def estimate_generation_max(nfe_max, start_pop_size, min_pop_size):
    """
    Number of generations that the linear reduction schedule will run.

    The article compares the generation index with Gmax / 2. With a shrinking
    population, Gmax is the number of generations whose evaluations fit into
    the remaining budget after initialization.
    """
    size = start_pop_size
    nfe = start_pop_size
    generations = 0
    while nfe < nfe_max:
        nfe += size
        generations += 1
        size = linear_population_size(nfe, nfe_max, start_pop_size, min_pop_size)
    return generations


def sinusoidal_decreasing(freq, generation, generation_max):
    """Non-adaptive sinusoidal decreasing adjustment, Eq. 4."""
    remaining = (generation_max - generation) / generation_max
    return 0.5 * (math.sin(2.0 * math.pi * freq * generation + math.pi) * remaining + 1.0)


def sinusoidal_increasing(freq, generation, generation_max):
    """Adaptive sinusoidal increasing adjustment, Eq. 5."""
    progress = generation / generation_max
    return 0.5 * (math.sin(2.0 * math.pi * freq * generation) * progress + 1.0)


def configuration_success_rate(successes, failures, epsilon=0.01):
    """Success rate of one sinusoidal configuration, Eq. 8. epsilon avoids a zero rate."""
    success_total = float(np.sum(successes))
    failure_total = float(np.sum(failures))
    total = success_total + failure_total
    if total == 0.0:
        return float(epsilon)
    return success_total / total + float(epsilon)


def configuration_probabilities(rates):
    """Probability of each sinusoidal configuration, Eq. 7."""
    rates = np.asarray(rates, dtype=float)
    return rates / np.sum(rates)


def weighted_lehmer_mean(values, deltas):
    """Weighted Lehmer mean used to update the parameter memory, Eqs. 13-15."""
    values = np.asarray(values, dtype=float)
    deltas = np.asarray(deltas, dtype=float)
    weights = deltas / np.sum(deltas)
    return np.sum(weights * np.square(values)) / np.sum(weights * values)


def euclidean_neighborhood_indices(points, best_index, fraction):
    """
    Indices of the NP * ps individuals closest to the best one.

    The best individual is included and returned first. The remaining places
    are filled by increasing Euclidean distance.
    """
    points = np.asarray(points, dtype=float)
    distances = np.linalg.norm(points - points[best_index], axis=1)
    count = round_half_up(len(points) * fraction)
    count = min(len(points), max(1, count))
    order = np.argsort(distances, kind="mergesort")
    neighbors = [int(index) for index in order if int(index) != int(best_index)]
    return [int(best_index)] + neighbors[:count - 1]


def covariance_eigenvectors(points):
    """
    Eigenvectors of the neighborhood covariance, Eq. 16.

    Columns of the returned basis are the eigenvectors, so C = B D B^T.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    dimension = points.shape[1]
    if points.shape[0] < 2:
        return np.eye(dimension)
    covariance = np.atleast_2d(np.asarray(np.cov(points, rowvar=False), dtype=float))
    if covariance.shape != (dimension, dimension):
        covariance = np.eye(dimension) * float(covariance.reshape(-1)[0])
    covariance = np.nan_to_num(covariance, nan=0.0)
    _, eigenvectors = np.linalg.eigh(covariance)
    return eigenvectors


def eigen_binomial_crossover(target, mutant, cr, basis):
    """
    Binomial crossover in the eigen coordinate system, Eqs. 17-20.

    Target and mutant are mapped by B^T, crossed, then mapped back by B.
    """
    target = np.asarray(target, dtype=float)
    mutant = np.asarray(mutant, dtype=float)
    target_eigen = basis.T @ target
    mutant_eigen = basis.T @ mutant
    trial_eigen = binomial_crossover_values(target_eigen, mutant_eigen, cr)
    return basis @ trial_eigen


def member_values(member):
    return [float(chromosome.real_value) for chromosome in member.chromosomes]
