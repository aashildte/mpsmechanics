# -*- coding: utf-8 -*-

"""

Performs operations relatd to the angular properties of given data.

Åshild Telle / Simula Research Laboratory / 2019

"""


import numpy as np

from .operations import calc_magnitude


def calc_projection(data, alpha):
    """

    Extracts the parallel part of each component in disp,
    with respect to given vector e_i (does a projection).

    Args:
        data - T x X x Y x 2 numpy array, original values
        alpha - relative to longitudial axis

    Returns:
        T x X x Y x 2 numpy array with parallel components

    """
    e_alpha = np.dot(np.array(((np.cos(alpha), -np.sin(alpha)),\
                               (np.sin(alpha), np.cos(alpha)))),\
                     np.array((1, 0)))
    f_dot = lambda x, _: np.dot(x, e_alpha)
    f_proj = lambda x, axis: np.apply_over_axes(f_dot, x, axis)

    return np.apply_over_axes(f_proj, data, -1)[:,:,:,0]
    

def calc_angle_diff(data, alpha):
    """

    Calculates the angle between a unit vector in direction based on
    the angle alpha and each vector in data.

    """
    e_alpha = np.dot(np.array(((np.cos(alpha), -np.sin(alpha)),\
                               (np.sin(alpha), np.cos(alpha)))),\
                     np.array((1, 0)))

    f_angle = lambda x, a : np.arccos(0) - np.arccos(np.abs(np.dot(x, e_alpha))/np.linalg.norm(x, axis=a))
    
    return np.nan_to_num(np.apply_over_axes(f_angle, data, -1))


def calc_projection_fraction(data, alpha):
    """

    Calculates fraction

        | projection(u, alpha)|
        --------------------------
                 || u ||

    of a given data set over time.

    Args:
        data - numpy array of dimensions T x X x Y x 2
        alpha - angle relative to longitudial axis

    Returns:
        T x X x Y x 2 numpy array, projection of normalized data

    """

    data_full = calc_magnitude(data)
    data_proj = np.abs(calc_projection(data, alpha))

    return np.divide(data_proj, data_full, \
            out=np.zeros_like(data_full), where=(data_full != 0))


def flip_values(data):
    """

    Rotate each vector, to first or fourth quadrant

    Args:
        data - T x X x Y x 2 numpy array, original values

    Returns:
        T x X x Y x 2 numpy array, flipped values

    """
    
    return np.where((data[:, :, :, 1] > 0)[:,:,:,None],
                    data, -data)
