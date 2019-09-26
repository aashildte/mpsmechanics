"""

Calculate force from displacement.

TODO
1) remove magical numbers
2) more descriptive names maybe? Eg F -> force

Åshild Telle / Simula Research Labratory / 2019

"""


import numpy as np


def displacement_to_force_area(delta_max, E, L, R, area):
    """
    all inputs in meters
    F_area #in mN/mm^2

    Args:
        delta_max - displacement, distance
        E - ?
        L - ?
        R - ?
        area - ? ; in mm^2

    Returns:
        Force per area
    """

    # I = 0.25 * np.pi * np.power(R, 4)
    # F = delta_max * (8*E*I) / (np.power(L, 3))
    return  displacement_to_force(delta_max, E, L, R) * 1000 / (area)


def displacement_to_force(delta_max, E, L, R):
    """
    all inputs in meters; returns force in N

    Args:
        ?

    Returns:
        ?

    """

    I = 0.25 * np.pi * np.power(R, 4)
    return delta_max * (8*E*I) / (np.power(L, 3))


def pxl_to_meters(x, scale):
    """

    Args:
        ?

    Returns:
        ?
    """

    return x * scale * 1e-6
