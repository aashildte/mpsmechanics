# -*- coding: utf-8 -*-

"""

Åshild Telle / Simula Research Labratory / 2019
David Cleres / UC Berkeley / 2019

"""


import numpy as np
from scipy import interpolate


def interpolate_values_2D(x_coords, y_coords, org_data):
    """

    Interpolates given data; defines functions based on this. The first
    function gives relative displacement (difference from first
    time frame); second absolute (relative to origo in given frame).

    Args:
        x_coords - x coordinates
        y_coords - y coordinates
        org_data - displacement data; X x Y x 2 numpy array

    Returns:
        function fn_abs : R2 - R2 - relative displacement
        function fn_rel : R2 - R2 - absolute displacement

    """

    # fn_x, fn_y - gives displacement in x / y direction

    fn_x = interpolate.interp2d(x_coords, y_coords, \
            org_data[:, :, 0].transpose(), kind='cubic')
    fn_y = interpolate.interp2d(x_coords, y_coords, \
            org_data[:, :, 1].transpose(), kind='cubic')

    fn_rel = lambda x, y: np.array([float(fn_x(x, y)), float(fn_y(x, y))])
    fn_abs = lambda x, y: np.array([x, y]) - fn_rel(x, y)

    return fn_abs, fn_rel
