# -*- coding: utf-8 -*-

"""

Åshild Telle / Simula Research Labratory / 2019

"""


import numpy as np
from scipy import interpolate


def interpolate_values_xy(x_coords, y_coords, org_data):
    """

    Interpolates given data; defines functions based on this. The first
    function gives relative displacement (difference from first
    time frame); second absolute (relative to origo in given frame).

    Args:
        x_coords - x coordinates
        y_coords - y coordinates
        org_data - displacement data; X x Y x 2 numpy array

    Returns:
        function fn_rel : R2 -> R2 - relative displacement
    """

    assert len(org_data.shape)==3, \
            "Error: Shape of input data not recognized."

    assert org_data.shape[-1] == 2, \
            "Error: Shape of input data not recognized."

    assert x_coords.shape[0] == org_data.shape[0], \
            "Error: Dimension mismatch of x_coords and org_data."

    assert y_coords.shape[0] == org_data.shape[1], \
            "Error: Dimension mismatch of y_coords and org_data."


    fn_x = interpolate.interp2d(x_coords, y_coords, \
            org_data[:, :, 0].transpose(), kind='cubic')
    fn_y = interpolate.interp2d(x_coords, y_coords, \
            org_data[:, :, 1].transpose(), kind='cubic')

    fn_rel = lambda x, y: np.array([float(fn_x(x, y)), float(fn_y(x, y))])

    return fn_rel
