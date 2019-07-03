# -*- coding: utf-8 -*-

"""

Functions to load displacement data.

Åshild Telle / Simula Research Labratory / 2019

"""

import numpy as np
from ..motion_tracking import motion_tracking as mt
from ..motion_tracking import ref_frame as rf
import mps

def read_mt_file(filename):
    """

    Passes on filename based on extension.

    Args:
        Filename - nd2 or csv file

    Returns:
        4-dimensional numpy array, of dimensions T x X x Y x 2
        scaling factor
        dimensions in x and y direction

    """

    assert (".nd2" in filename or ".csv" in filename), \
            "Unknown file formate"

    if ".nd2" in filename:
        return _read_file_nd2(filename)
    return _read_file_csv(filename)      # if not nd2, then csv


def _read_file_nd2(filename):
    """
    Gets displacement from the mt module.

    Args:
        filename - nd2 file

    Returns:
        4-dimensional numpy array, of dimensions T x X x Y x 2
        scaling factor - um per pixel
        dimensions - number of macroblocs in x and y directions

    """
    mt_data = mps.MPS(filename)
    scaling_factor = mt_data.info['um_per_pixel']
    dimensions = mt_data.frames.shape[:-1]
    motion = mt.MotionTracking(mt_data, use_cache=True)

    # get right reference frame

    data_disp = motion.displacement_vectors

    # different conventions

    ref_fn = rf.calculate_min_velocity_frame
    #ref_fn = rf.calculate_minimum_2step
    
    data_disp = rf.convert_disp_data(data_disp, ref_fn(data_disp))

    # convert to T x X x Y x 2 TODO maybe we can do this in
    # motiontracking actually

    data_disp = np.swapaxes(np.swapaxes(np.swapaxes(\
            data_disp, 0, 1), 0, 2), 0, 3)

    return data_disp, scaling_factor, dimensions


def _read_file_csv(filename):
    """

    Reads the input file, where the file is assumed to be a csv file on
    the form

        T, X, Y, scaling factor, dimension x dir., dimension y dir.
        x0, y0
        x1, y1
        ...

    where we have T x X x Y values, giving the position for a given unit
    for time step t0, t1, ..., for  x coordinates x0, x1, ... and
    y coordinates y0, y1, ...

    x0, y0 gives relative motion of unit (0, 0) at time step 0; x1, y1
    gives relative motion of unit (1, 0) at time step 0, etc.

    Args:
        filename - csv file

    Returns:
        4-dimensional numpy array, of dimensions T x X x Y x 2
        scaling factor
        dimensions in x and y direction

    """

    f = open(filename, 'r')

    T, X, Y, scaling_factor, dims_x, dims_y = f.readline().split(",")
    T, X, Y = map(int, (T, X, Y))
    scaling_factor, dims_x, dims_y = \
            map(float, (scaling_factor, dims_x, dims_y))
    data = np.zeros((T, X, Y, 2))

    for t in range(T):
        for i in range(X):
            for j in range(Y):
                str_values = str.split(f.readline().strip(), ",")
                disp = list(map(float, str_values))
                data[t, i, j] = np.array(disp)

    f.close()

    return data, scaling_factor, np.array((dims_x, dims_y))
