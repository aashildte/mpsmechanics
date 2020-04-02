"""

Calculate force from displacement.

Åshild Telle / Simula Research Labratory / 2020

"""

import numpy as np


def displacement_to_force(
    displacement: np.ndarray,
    elastic_modulus: float,
    height: float,
    radius: float,
):
    """

    Calculates force at each pillar, in N.

    Args:
        displacement - how much the pillar has moved, in um; numpy array
            of dimension num_time_steps x num_pillars x 2
        elastic_modulus - given by material properties; in ???
        height - height of pillar, in um
        radius - radius of pillar, in um

    Returns:
        force calculated for each entry in the array; numpy array of dimensions
            num_time_steps x num_pillars x 2, in N

    """

    interia_moment = 0.25 * np.pi * radius ** 4

    return displacement * (8 * elastic_modulus * interia_moment) / height ** 3


def displacement_to_force_area(
    force: np.ndarray, height: float, radius: float
):
    """

    Calculates force per area at each pillar, in mN / mm^2.

    Args:
        force - force calculated for pillar, in N
        height - height of pillar, in um
        radius - radius of pillar, in um

    Returns:
        force per area, in mN / mm^2
    """

    area = height * radius * np.pi * 1e6  # mm^2
    force_mN = 1000 * force

    return force_mN / area
