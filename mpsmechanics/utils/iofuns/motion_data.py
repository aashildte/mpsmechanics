# -*- coding: utf-8 -*-

"""

Functions to load displacement data.

Åshild Telle / Simula Research Labratory / 2019

"""

import numpy as np
import mps

from ...motion_tracking import motion_tracking as mt
from ...motion_tracking import ref_frame as rf

def read_mt_file(filename, method):
    """

    Passes on filename based on extension.

    Args:
        Filename - nd2 or csv file

    Returns:
        4-dimensional numpy array, of dimensions T x X x Y x 2
        scaling factor
        dimensions in x and y direction

    """

    assert (".nd2" in filename or ".npy" in filename), \
            "Unknown file formate"

    if ".nd2" in filename:
        return _read_file_nd2(filename, method)

    print("TODO : Implement npy file formate.")


def _read_file_nd2(filename, method=None):
    """
    Gets displacement from the mt module.

    Args:
        filename - nd2 file

    Returns:
        4-dimensional numpy array, of dimensions T x X x Y x 2
        scaling factor - um per pixel
        dt - time step (ms)
        angle - angle correction

    """
    mt_data = mps.MPS(filename)

    scaling_factor = mt_data.info['um_per_pixel']
    motion = mt.MotionTracking(mt_data, use_cache=True)

    # get right reference frame

    data_disp = motion.displacement_vectors
    angle = motion.angle

    # different conventions

    if method=="velocity":
        ref_fn = rf.calculate_min_velocity_frame
    elif method=="minmax":
        ref_fn = rf.calculate_minimum_2step
    elif method=="firstframe":
        ref_fn = rf.calculate_firstframe
    elif method=="mean":
        pass
    else:
        print("Error: Method not recognized")
        exit(-1)

    if method != "mean":
        data_disp = rf.convert_disp_data(data_disp, ref_fn(data_disp))

    # convert to T x X x Y x 2 TODO maybe we can do this in
    # motiontracking actually

    data_disp = np.swapaxes(np.swapaxes(np.swapaxes(\
            data_disp, 0, 1), 0, 2), 0, 3)

    return data_disp, scaling_factor, angle, \
            mt_data.dt, mt_data.size_x, mt_data.size_y      # maybe??
