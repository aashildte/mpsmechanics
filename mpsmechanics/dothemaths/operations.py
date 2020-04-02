# -*- coding: utf-8 -*-

"""

Åshild Telle / Simula Research Labratory / 2019

"""


import numpy as np


def calc_norm_over_time(data):
    """
    Finds sum of norm of each component

    Args:
        Data - numpy array, of dimensions T x X x Y x D
        where D can be (), (2,) or (2, 2)

    Returns:
        Sum array - numpy array of dimension T

    """
    assert data.shape[3:] in (
        (2, 2),
        (2,),
        (),
    ), "Error: Shape of input data not recognized."

    return np.sum(calc_magnitude(data), axis=(1, 2))


def calc_magnitude(data):
    """

    Get the norm of the vector for every point (x, y).

    Args:
        data - numpy array of dimension T x X x Y x D

    Returns:
        norm of data - numpy array of dimension T x X x Y

    """

    assert data.shape[3:] in (
        (2, 2),
        (2,),
        (),
    ), "Error: Shape of input data not recognized."

    if data.shape[3:] == (2, 2):
        magnitude = np.linalg.norm(data, axis=(3, 4), ord=2)
    elif data.shape[3:] == ():
        magnitude = np.abs(data)
    else:  # then (2)
        magnitude = np.linalg.norm(data, axis=3)

    return magnitude


def normalize_values(data):
    """

    Normalises each non-zero vector.

    Args:
        data - T x X x Y x D numpy array, original values

    Returns:
        T x X x Y x 2 numpy array, normalized values

    """

    assert data.shape[3:] in (
        (2, 2),
        (2,),
        (),
    ), "Error: Shape of input data not recognized."

    return np.divide(
        data,
        calc_magnitude(data)[:, :, :, None],
        out=np.zeros_like(data),
        where=data != 0,
    )
