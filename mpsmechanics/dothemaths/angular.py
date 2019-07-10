# -*- coding: utf-8 -*-

"""

Performs operations relatd to the angular properties of given data.

Åshild Telle / Simula Research Laboratory / 2019

"""


import os
import numpy as np
import matplotlib.pyplot as plt

from . import operations as op


def calc_projection_vectors(data, e_i):
    """

    Extracts the parallel part of each component in disp,
    with respect to given vector e_i (does a projection).

    Args:
        data - (T x) X x Y x 2 numpy array, original values
        e_i - non-zero unit vector (numpy array/two values)

    Returns:
        (T x) X x Y x 2 numpy array with parallel components

    """

    f = lambda x, axis: np.dot(x, e_i)
    g = lambda x, axis: np.apply_over_axes(f, x, axis)*e_i

    return np.apply_over_axes(g, data, -1)


def calc_projection_values(data, angle):
    """

    Calculates angular fraction of a given data set over time.

    Args:
        data - numpy array of dimensions T x X x Y x 2
        angle - how much the chamber is tilted

    Returns:
        x projection (longest direction) of normalized data

    """

    c, s = np.cos(angle), np.sin(angle)
    e_alpha = np.dot(np.array(((c, -s), (s, c))), np.array((1, 0)))
    
    data_full = np.linalg.norm(data, axis=-1)
    data_xdir = np.linalg.norm(calc_projection_vectors(data, \
            e_alpha), axis=-1)
    
    return np.divide(data_xdir, data_full, \
            out=np.zeros_like(data_full), where=data_full!=0)


def flip_values(data, over_time):
    """

    Rotate each vector, to first or fourth quadrant

    Args:
        data - (T x) X x Y x 2 numpy array, original values
        over_time - boolean value; determines if T dimension
            should be included or not

    Returns:
        (T x) X x Y x 2 numpy array, flipped values

    """

    f = lambda x, i, j: -x if x[1] < 0 else x

    return op.perform_operation(data, f, over_time=over_time)
