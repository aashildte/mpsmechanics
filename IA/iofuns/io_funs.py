"""

Functions for IO operations: Reading files, creating folder structures.

Åshild Telle / Simula Research Labratory / 2019

"""


import sys
import os
import numpy as np
import matplotlib.pyplot as plt

import mps


def read_file(filename, xlen):

    if(".nd2" in filename):
        return read_file_nd2(filename, xlen)
    elif(".csv" in filename):
        return read_file_csv(filename, xlen)
    else:
        print("Uknown file formate")
        exit(-1)


def read_file_nd2(filename, xlen):

    mps_data = mps.MPS(filename)
    motion = mps.MotionTracking(mps_data)
    motion.run()
    motion.GetContractionData(datatype="Disp")
    data = motion.results["MotionBioFormatsStrainInterval"]["motionVect"]

    # reshape; t as outer dimension - or??

    X, Y, D, T = data.shape

    dx = xlen/X

    data_disp = np.zeros((T, X, Y, D))

    for t in range(T):
        for x in range(X):
            for y in range(Y):
                for d in range(D):
                    data_disp[t, x, y, d] = dx*data[x, y, d, t]

    scale = _get_scale(data_disp)

    return scale*data_disp, 1./scale


def read_file_csv(filename, xlen):
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
        xlen - length of picture, in meters

    Returns:
        4-dimensional numpy array, of dimensions T x X x Y x 2
        scale - scale data points with this to get original magnitude

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

    # omit first value; reference configuration

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


def get_path(filename):
    """

    Remove relative paths + file suffix

    """

    # strip f_in for all relative paths

    while(".." in filename):
        r_ind = filename.find("..") + 3
        filename = filename[r_ind:]

    # and for file type / suffix

    r_ind = filename.find(".")
    filename = filename[:r_ind]

    return filename


def get_idt(filename):
    """

    Strip path + file suffix

    """

    filename = os.path.normpath(filename).split(os.path.sep)[-1]
    return filename.split(".")[0]


def make_dir_structure(path):
    """

    Makes a directory structure based on a given path.

    """

    dirs = os.path.normpath(path).split(os.path.sep)
    
    acc_d = ""

    for d in dirs:
        acc_d = os.path.join(acc_d, d)
        if not (os.path.exists(acc_d)):
            os.mkdir(acc_d)


if __name__ == "__main__":
    
    try:
        f_in = sys.argv[1]
    except:
        print("Give file name as first argument")
        exit(-1)

    assert(read_file_csv(f_in, 1) is not None)
    make_dir_structure("Figures")         # no return value

    print("All tests passed for io_funs.py")

