"""

Functions for IO operations: Reading files, creating folder structures.

Åshild Telle / Simula Research Labratory / 2019

"""


import sys
import os
import numpy as np
import matplotlib.pyplot as plt

import mps


def read_file(filename):
    if(".nd2" in filename):
        return _read_file_nd2(filename)
    elif(".csv" in filename):
        return _read_file_csv(filename)
    else:
        print("Uknown file formate")
        exit(-1)


def _read_file_nd2(filename):

    mps_data = mps.MPS(filename)
    motion = mps.MotionTracking(mps_data)
    motion.run()
    motion.GetContractionData(datatype="Disp")
    data = motion.results["MotionBioFormatsStrainInterval"]["motionVect"]

    # reshape; t as outer dimension - or??

    X, Y, D, T = data.shape
    data_disp = np.zeros((T, X, Y, D))

    for t in range(T):
        for x in range(X):
            for y in range(Y):
                for d in range(D):
                    data_disp[t, x, y, d] = data[x, y, d, t]

    return data_disp


def _read_file_csv(filename):
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

    The displacement is given as *scaled*, i.e. scaled to a range around 0.
    This is to avoid potential problems with numerical precision.

    Args:
        filename - csv file

    Returns:
        4-dimensional numpy array, of dimensions T x X x Y x 2

    """

    f = open(filename, 'r')

    T, X, Y = map(int, str.split(f.readline(), ","))
    data = np.zeros((T, X, Y, 2))
    
    for t in range(T):
        for i in range(X):
            for j in range(Y):
                str_values = str.split(f.readline().strip(), ",")
                d = list(map(float, str_values))
                data[t, i, j] = np.array(d)
    
    f.close()

    # omit first value; reference configuration

    return data
