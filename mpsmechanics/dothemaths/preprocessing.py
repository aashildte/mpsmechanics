# -*- coding: utf-8 -*-

"""

Given experimental data on movement this function preprocess the
given values. This is interesting if the input values are coarse,
come with a low sampling resolution; this algorithm will give you
a smoother vector field.

Åshild Telle / Simula Research Labratory / 2019
David Cleres / UC Berkeley / 2019

"""

import numpy as np
import cv2


def _diffusion_step(data, alpha, N_diff=1):
    """

    Do diffusion/averaging of values using the molecule
          x
        x o x
          x
    weighting the middle point with value alpha and the surrounding
    points with value (1 - alpha).

    Boundary points are using molecule on the form

        x 0 x
          x

    or

        x 0
          x
    """

    factor = 0.25
    neighbour_weight = (1 - alpha) * factor

    kernel = np.array([[0, neighbour_weight, 0], [neighbour_weight, alpha, \
            neighbour_weight], [0, neighbour_weight, 0]])
    dst = data.copy()

    for _ in range(N_diff):
        dst[:, :, 0] = cv2.filter2D(dst[:, :, 0], -1, kernel)
        dst[:, :, 1] = cv2.filter2D(dst[:, :, 1], -1, kernel)

    return dst


def do_diffusion(data, alpha, N_diff, over_time):
    """
    Performs a moving averaging algorithm of given data, using
    auxilary function _diffusion_step.

    Args:
        data - numpy array of dimensions (T x) X x Y x 2
        alpha - weight, value between 0 and 1
        N_diff - number of times to run diffusion
        over_time - boolean value; determines if T dimension
            should be included or not

    Returns:
        (T x) X x Y x 2 numpy array, data after averaging process

    """

    if over_time:
        T = data.shape[0]
        new_data = np.zeros_like(data)

        for t in range(T):
            new_data[t] = _diffusion_step(data[t], alpha, N_diff)
    else:
        new_data = _diffusion_step(data, alpha, N_diff)

    return new_data


def calc_filter(data, threshold):

    T, X, Y = data.shape[:3]

    movement = np.full((X, Y), False)

    for t in range(T):
        for x in range(X):
            for y in range(Y):
                if np.linalg.norm(data[t, x, y]) >= threshold:
                    movement[x, y] = True

    return movement
