from enum import Enum
import numpy as np
from detpy.models.member import Member
from detpy.models.population import Population


class BoundaryFixing(Enum):
    CLIPPING = 'clipping'
    REFLECTION = 'reflection'
    REFLECTION_BACK = 'reflection_back'
    RANDOM = 'random'
    WITH_PARENT = 'with_parent'


def get_boundary_constraints_fun(fix_type: BoundaryFixing):
    return {
        BoundaryFixing.CLIPPING: lambda member: boundary_clipping(member),
        BoundaryFixing.REFLECTION: lambda member: boundary_reflection(member),
        BoundaryFixing.REFLECTION_BACK: lambda member: boundary_reflection_back(member),
        BoundaryFixing.RANDOM: lambda member: boundary_random(member),
        BoundaryFixing.WITH_PARENT: lambda trial, parent: boundary_with_parent(trial, parent),
    }.get(fix_type, lambda: None)


def fix_boundary_constraints(population: Population, fix_type: BoundaryFixing):
    boundary_constraints_fun = get_boundary_constraints_fun(fix_type)
    if fix_type == BoundaryFixing.WITH_PARENT:
        raise ValueError("fix_type=WITH_PARENT requires a trial population.")
    else:
        for member in population.members:
            if not member.is_member_in_interval():
                boundary_constraints_fun(member)


def fix_boundary_constraints_with_parent(population: Population, trial: Population,
                                         fix_type: BoundaryFixing, ):
    boundary_constraints_fun = get_boundary_constraints_fun(fix_type)

    if fix_type == BoundaryFixing.WITH_PARENT:
        for member_trial, member_parent in zip(trial.members, population.members):
            boundary_constraints_fun(member_trial, member_parent)
    else:
        for member in trial.members:
            if not member.is_member_in_interval():
                boundary_constraints_fun(member)


# Strategies for fixing members, when they are beyond boundaries


def boundary_clipping(member: Member):
    """
    Modifies the values of `member` in-place.

    :param member: The member to be modified.
    """
    for chromosome in member.chromosomes:
        if chromosome.real_value > chromosome.ub:
            chromosome.real_value = chromosome.ub
        elif chromosome.real_value < chromosome.lb:
            chromosome.real_value = chromosome.lb


def boundary_reflection(member: Member):
    """
    Modifies the values of `member` in-place.

    :param member: The member to be modified.
    """
    for chromosome in member.chromosomes:
        if chromosome.real_value > chromosome.ub:
            chromosome.real_value = 2 * chromosome.ub - chromosome.real_value
        elif chromosome.real_value < chromosome.lb:
            chromosome.real_value = 2 * chromosome.lb - chromosome.real_value


def boundary_random(member: Member):
    """
    Modifies the values of `member` in-place.

    :param member: The member to be modified.
    """
    for chromosome in member.chromosomes:
        if chromosome.real_value < chromosome.lb or chromosome.real_value > chromosome.ub:
            chromosome.real_value = np.random.uniform(chromosome.lb, chromosome.ub)


def boundary_reflection_back(member: Member):
    """
    Modifies the values of `member` in-place.

    :param member: The member to be modified.
    """
    for chromosome in member.chromosomes:
        range_i = chromosome.ub - chromosome.lb
        if chromosome.real_value > chromosome.ub:
            chromosome.real_value = chromosome.ub - (chromosome.real_value - chromosome.ub) + int(
                (chromosome.real_value - chromosome.ub) / range_i) * range_i
        elif chromosome.real_value < chromosome.lb:
            chromosome.real_value = chromosome.lb - (chromosome.lb - chromosome.real_value) + int(
                (chromosome.lb - chromosome.real_value) / range_i) * range_i


def boundary_with_parent(member_trial: Member, member_parent: Member):
    """
    Modifies the values  'member' relative to its parent.

    :param member_trial: The member to be modified.
    :param member_parent: The parent member used as a reference.
    """
    for chromosome_trial, chromosome_parent in zip(member_trial.chromosomes, member_parent.chromosomes):
        if chromosome_trial.real_value > chromosome_trial.ub:
            chromosome_trial.real_value = (chromosome_trial.ub + chromosome_parent.real_value) / 2
        elif chromosome_trial.real_value < chromosome_trial.lb:
            chromosome_trial.real_value = (chromosome_trial.lb + chromosome_parent.real_value) / 2
