"""

Module for IO operations

Åshild Telle / Simula Research Labratory / 2019

"""


import sys
import os
import numpy as np
import matplotlib.pyplot as plt


def read_disp_file(filename, xlen):
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

    Arguments:
        filename - csv file
        xlen - length of picture, in m

    Returns:
        4-dimensional numpy array, of dimensions T x X x Y x 2
        scale - how much to scale output with in the end

    """

    f = open(filename, 'r')

    T, X, Y = map(int, str.split(f.readline(), ","))

    dx = xlen/X

    data = np.zeros((T, X, Y, 2))

    for t in range(T):
        for i in range(X):
            for j in range(Y):
                str_values = str.split(f.readline().strip(), ",")
                d = list(map(float, str_values))
                data[t, i, j] = dx*np.array(d)
    
    f.close()

    scale = _get_scale(data)

    return scale*data, 1./scale


def _get_scale(data):
    """

    Scales data such that the middle value is on standard form, i.e.
    in the range [0.1, 1]

    """


    # get maximum and minimum; scale middle value

    mmax = abs(np.max(data))
    mmin = abs(np.min(data))

    mid = 0.5*(mmax + mmin)

    # scale from above or below; dependin on scale

    scale = 1

    while(mid > 1):
        mid = mid/10
        scale = scale/10

    while(mid < 1E-1):
        mid = mid*10
        scale = scale*10

    return scale


def get_os_del():
    return "\\" if os.name=="nt" else "/"

def make_dir_structure(path):
    """

    Makes a directory structure based on a given path.

    """

    # Folder structure different depending on OS,
    # check and assign different for Windows and Linux/Mac
    
    de = get_os_del()

    dirs = path.split(de)

    acc_d = "."

    for d in dirs:
        acc_d = acc_d + de + d
        if not (os.path.exists(acc_d)):
            os.mkdir(acc_d)


if __name__ == "__main__":
    
    try:
        f_in = sys.argv[1]
    except:
        print("Give file name as first argument")
        exit(-1)

    data, scale = read_disp_file(f_in, 1)

    assert(len(data.shape)==4)

    print("Read data test passed")
    print("All tests passed")

