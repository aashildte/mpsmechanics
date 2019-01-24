"""

Module for IO operations

Åshild Telle / Simula Research Labratory / 2019

"""


import sys
import os
import numpy as np
import matplotlib.pyplot as plt



def read_disp_file(filename):
    """

    Reads the input file, where the file is assumed to be a csv file on
    the form
            T, X, Y
            x0, y0
            x1, y1
            ...
    where we have T x X x Y values, giving the position for a given unit
    for time step t0, t1, ..., for  x coordinates x0, x1, ... and
    y coordinates y0, y1, ...

    x0, y0 gives relative motion of unit (0, 0) at time step 0; x1, y1
    gives relative motion of unit (1, 0) at time step 0, etc.

    Arguments:
        filename - csv file

    Returns:
        4-dimensional numpy array of rec_dimensions
            T x X x Y x 2

    """

    f = open(filename, 'r')
    T, X, Y = map(int, str.split(f.readline(), ","))

    data = np.zeros((T, X, Y, 2))

    for t in range(T):
        for i in range(X):
            for j in range(Y):
                d = str.split(f.readline(), ",")[:2]
                data[t, i, j] = np.array(d)

    f.close()

    return data

