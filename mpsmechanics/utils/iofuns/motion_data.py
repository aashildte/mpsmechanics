# -*- coding: utf-8 -*-

"""

Functions to load displacement data.

TODO depricated??

Åshild Telle / Simula Research Labratory / 2019

"""

import numpy as np
import mps


from .data_layer import read_prev_layer
from ...motion_tracking.motion_tracking import track_motion

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


def _read_file_nd2(filename, method):
    """
    Gets displacement from the mt module.

    Args:
        filename - nd2 file

    Returns:
        4-dimensional numpy array, of dimensions T x X x Y x 2
        scaling factor - um per pixel
        angle - angle correction
        dt - time step (ms)
        size_x
        size_y
    """

    save_data = True
    layer_name = "track_motion_" + method
    layer_fn = lambda f_in, save_data, method=method: track_motion(f_in, method, save_data=save_data)

    data = read_prev_layer(filename, layer_name, layer_fn, save_data)

    return data["data_disp"], data["scaling_factor"], data["angle"], \
            data["dt"], data["size_x"], data["size_y"]
