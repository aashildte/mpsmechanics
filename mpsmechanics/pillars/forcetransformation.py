"""

Calculate force from displacement.

David Cleres / UC Berkeley / 2019
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
        area - ?

    Returns:
        ?
    """

    I = 0.25 * np.pi * np.power(R,4)
    F = delta_max * (8*E*I) / (np.power(L, 3))
    return  F * 1000 / (area) #area is already in mm^2 


def displacement_to_force(delta_max, E, L, R):
    """
    all inputs in meters 
    F return force in N 

    Args:
        ?

    Returns:
        ?

    """

    I = 0.25 * np.pi * np.power(R,4)
    return delta_max * (8*E*I) / (np.power(L, 3))


def pxl_to_meters(x, scale):
    """

    Args:
        ?

    Returns:
        ?
    """
    
    return x * scale * 1e-6
