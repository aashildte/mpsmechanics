"""

Performs operations relatd to the angular properties of given data.

Åshild Telle / Simula Research Labratory / 2019

"""


import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

import dothemaths.operations as op

def calc_direction_vectors(disp, plt_pr, mu=1E-14):
    """

    From the given displacement, this function finds the
    direction of most detected movement using linear regression.

    If specified in plt_pr, this function calls another
    function which plots the values along with direction vectors for 
    a visual check.

    Arguments:
        disp - T x X x Y x 2 dimensional numpy array
        plt_pr - directory for plotting options
        mu - optional argument, threshold for movement detection

    Returns:
	e_alpha - vector in 1st or 4th quadrant along most movement
	e_beta  - e_alpha rotatet pi/2 anti-clockwise

    """
    T, X, Y = disp.shape[:3]

    xs = []
    ys = []

    for t in range(T): 
        for x in range(X):
            for y in range(Y):
                if(np.linalg.norm(disp[t, x, y]) >= mu):
                    xs.append(x)
                    ys.append(y)

    slope = st.linregress(xs, ys)[0]

    dir_v = np.array([1, slope])

    e_alpha = np.linalg.norm(dir_v)*dir_v
    e_beta  = np.array([-e_alpha[1], e_alpha[0]])
    
    if(plt_pr["visual check"]):
        dimensions = plt_pr["dims"]
        _plot_data_vectors(xs, ys, X, Y, e_alpha, e_beta,
                plt_pr["idt"], dimensions)
    
    return e_alpha, e_beta


def _plot_data_vectors(xs, ys, X, Y, e_alpha, e_beta, idt, dimensions):
    """

    Plots data points along with direction vectors.

    Figures saved as
        idt + _alignment.png
    in a folder called Figures

    Arguments:
        xs - data points along x axis
        ys - data points along y axis
        X  - number of data points in x direction
        Y  - number of data points in y direction
        e_alpha - main direction
        e_beta  - perpendicular vector
        idt - idt for plots
        dimensions - pair of dimensions (x, y)

    """
    # scale dimensions to standard size in x direction

    scale = 6.4/dimensions[0]
    dimensions_scaled = (scale*dimensions[0], scale*dimensions[1])

    plt.figure(figsize=dimensions_scaled)
    plt.xlabel('$\mu m$')
    plt.ylabel('$\mu m$')

    # downsample data for plotting - only plot each value once

    pairs = list(set([(x, y) for (x, y) in zip(xs, ys)]))

    p_x = np.array([p[0] for p in pairs])
    p_y = np.array([p[1] for p in pairs])

    # scale

    p_x = dimensions[0]/X*p_x
    p_y = dimensions[1]/Y*p_y

    plt.scatter(p_x, p_y, color='gray')

    sc = [0.1*max(p_x), 0.1*max(p_y)]

    for e in [e_alpha, e_beta]:
        plt.plot([0, sc[0]*e[0]], [0, sc[1]*e[1]], color='red')

    de = io.get_os_delimiter()
    path = de.join(("Figures" + de + idt).split(de)[:-1])
    io.make_dir_structure(path)

    plt.savefig("Figures" + de + idt + "_alignment.png", dpi=1000)

    plt.figure()
    plt.clf()


def calc_projection_vectors(data, e_i, over_time):
    """

    Extracts the parallel part of each component in disp,
    with respect to given vector e_i (does a projection).

    Arguments:
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


    Arguments:
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
                if(norm > 1E-14):
                    data_x[t, x, y] = data_x[t, x, y]/norm
                else:
                    data_x[t, x, y] = np.zeros(2)
    
    return data_x


def flip_values(data, over_time):
    """

    Rotate each vector, to first or fourth quadrant

    Arguments:
        data - (T x) X x Y x 2 numpy array, original values
        over_time - boolean value; determines if T dimension
            should be included or not

    Returns:
        (T x) X x Y x 2 numpy array, flipped values

    """

    f = lambda x, i, j: -x if x[1] < 0 else x

    return op.perform_operation(data, f, over_time=over_time)


if __name__ == "__main__":

    # unit tests

    data = np.random.rand(3, 3, 3, 2)

    ppl = {}
    for i in range(8):
        ppl[int(i)] = {"plot" : False}
    
    ppl["idt"] = "unit_tests"
    ppl["visual check"] = False
    ppl["dims"] = (6, 4)

    # unit tests:
    
    e_a, e_b = calc_direction_vectors(data, ppl)

    assert((e_a, e_b) is not None)
    assert(calc_projection_vectors(data, e_a, over_time=True) \
            is not None)
    assert(flip_values(data, over_time=True) is not None)
    assert(calc_projection_values(data, e_a) is not None)
    
    print("All checks passed for angular.py")
