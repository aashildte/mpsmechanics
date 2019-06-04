# -*- coding: utf-8 -*-

"""

Performs operations relatd to the angular properties of given data.

Åshild Telle / Simula Research Labratory / 2019

"""


import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

from . import operations as op

def calc_direction_vectors(disp, plt_pr, movement_filter):
    """

    From the given displacement, this function finds the direction
    of most detected movement using linear regression.

    If specified in plt_pr, this function calls another function which
    plots the values along with direction vectors for a visual check.

    Args:
        disp - T x X x Y x 2 dimensional numpy array
        plt_pr - directory for plotting options
        movement_filter - boolean numpy array of size X x Y

    Returns:
	e_alpha - vector in 1st or 4th quadrant along most movement
	e_beta  - e_alpha rotated pi/2 anti-clockwise

    """

    _, X, Y, _ = disp.shape

    xs = []
    ys = []

    for x in range(X):
        for y in range(Y):
            if(movement_filter[x, y]):
                xs.append(x)
                ys.append(y)

    xs, ys = np.array(xs), np.array(ys)

    slope = st.linregress(xs, ys)[0]

    dir_v = np.array([1, slope])

    e_alpha = 1./np.linalg.norm(dir_v)*dir_v
    e_beta = np.array([-e_alpha[1], e_alpha[0]])

    if plt_pr["visual check"]:
        _plot_data_vectors(xs, ys, X, Y, e_alpha, e_beta, plt_pr)

    return e_alpha, e_beta


def _plot_data_vectors(xs, ys, X, Y, e_alpha, e_beta, plt_pr):
    """

    Plots data points along with direction vectors.

    Figures saved as
        idt + _alignment.png
    in a folder called Figures

    Args:
        xs - data points along x axis
        ys - data points along y axis
        X  - number of data points in x direction
        Y  - number of data points in y direction
        e_alpha - main direction
        e_beta  - perpendicular vector
        plt_pr - plotting properties dictionary

    """
    # scale dimensions to standard size in x direction

    dimensions = plt_pr["dims"]
    scale = 6.4/dimensions[0]
    dimensions_scaled = (scale*dimensions[0], scale*dimensions[1])

    eps_x = 0.05*dimensions[0]
    eps_y = 0.05*dimensions[1]

    plt.figure(figsize=dimensions_scaled)
    plt.xlim(-eps_x, dimensions[0] + eps_x)
    plt.ylim(-eps_y, dimensions[1] + eps_y)
    plt.xlabel('$\mu m$')
    plt.ylabel('$\mu m$')

    # scale
    p_x = dimensions[0]/X*xs
    p_y = dimensions[1]/Y*ys

    plt.scatter(p_x, p_y, color='gray')

    sc = 0.15*max(p_x)

    for e in [e_alpha, e_beta]:
        plt.arrow(eps_x/2, eps_y/2, sc*e[0], sc*e[1], \
                width=0.005*dimensions[0], color='red')

    plt.savefig(os.path.join(plt_pr["path"], "alignment.png"), dpi=1000)
    plt.close()

def calc_projection_vectors(data, e_i, over_time):
    """

    Extracts the parallel part of each component in disp,
    with respect to given vector e_i (does a projection).

    Args:
        data - (T x) X x Y x 2 numpy array, original values
        e_i - non-zero vector (numpy array/two values)
        over_time - boolean value; determines if T dimension
            should be included or not

    Returns:
        (T x) X x Y x 2 numpy array with parallel components

    """
    e_i = 1./(np.linalg.norm(e_i))*e_i

    f = lambda x, i, j, e_i=e_i: np.dot(x, e_i)*e_i

    return op.perform_operation(data, f, over_time=over_time)


def calc_projection_values(data, e_alpha):
    """

    Calculates angular fraction of a given data set over time.


    Args:
        data - numpy array of dimensions T x X x Y x 2
        e_alpha - defining direction of interest

    Returns:
        x fraction of data over time

    """

    data_x = calc_projection_vectors(data, e_alpha, over_time=True)
    T, X, Y = data.shape[:3]

    # replaces data in data_x

    for t in range(T):
        for x in range(X):
            for y in range(Y):
                norm = np.linalg.norm(data[t, x, y])
                if norm > 1E-14:
                    data_x[t, x, y] = data_x[t, x, y]/norm
                else:
                    data_x[t, x, y] = np.zeros(2)

    return data_x


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
