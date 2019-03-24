"""

Given experimental data on movement this function preprocess the
given values. This is interesting if the input values are coarse,
come with a low sampling resolution; this algorithm will give you
a smoother vector field.

Åshild Telle / Simula Research Labratory / 2019

"""


import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

import io_funs as io


def _diffusion_step(data, alpha, N_diff):
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
    
    # diffusion/averaging molecule

    m = lambda a, i, j : alpha*a[i][j] + 0.25*(1 - alpha)* \
             (a[max(i-1,0)][j] + a[i][max(j-1,0)] +
             a[min(i+1,X-1)][j] + a[i][min(j+1,Y-1)])

    X, Y = data.shape[:2]

    d1 = data.copy()
    d2 = np.zeros_like(data)

    for n in range(N_diff):
        for i in range(X):
           for j in range(Y):
               d2[i][j] = m(d1, i, j)

        d1, d2 = d2, d1       # swap pointers

    return d1


def do_diffusion(data, alpha, N_diff, over_time=True):
    """
    Performs a moving averaging algorithm of given data, using
    auxilary function _diffusion_step.

    Arguments:
        data - numpy array of dimensions (T x) X x Y x 2
        alpha - weight, value between 0 and 1
        N_diff - number of times to run diffusion
        over_time - boolean value; determines if T dimension
            should be included or not

    Returns:
        (T x) X x Y x 2 numpy array, data after averaging process

    """

    if(over_time):
        T = data.shape[0]
        new_data = np.zeros_like(data)

        for t in range(T):
            new_data[t] = _diffusion_step(data[t], alpha, N_diff)
    else:
        new_data = _diffusion_step(data, alpha, N_diff)

    return new_data



if __name__ == "__main__":

    try:
        f_in = sys.argv[1]
    except:
        print("Error: Give displacement file name as first positional argument.")
        exit(-1)
    
    data, scale = io.read_disp_file(f_in, 1)

    # unit tests:
    
    assert(do_diffusion(data, 0.75, 2) is not None)

    print("All checks passed for preprocessing.py")
