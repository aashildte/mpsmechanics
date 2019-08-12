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


def read_mt_file(filename):
    """

    Passes on filename based on extension.

    Args:
        Filename - nd2 or csv

    Returns:
        4-dimensional numpy array, of dimensions T x X x Y x 2
        scaling factor
        dimensions in x and y direction

    """

    assert (".nd2" in filename or ".csv" in filename), \
            "Unknown file formate"

    if ".nd2" in filename:
        return _read_file_nd2(filename)

    print("TODO : Implement csv file formate.")


def _read_file_nd2(filename):
    """
    Gets displacement from motion tracking layer ... review this function

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

    print("hei??")

    layer_name = "track_motion"
    
    print("hei??")

    data = read_prev_layer(filename, layer_name, track_motion, save_data)
    
    print("hei??")

    return data["data_disp"], data["scaling_factor"], data["angle"], \
            data["dt"], data["size_x"], data["size_y"]
