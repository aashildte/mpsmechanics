"""

Åshild Telle / Simula Research Labratory / 2019
David Cleres / UC Berkeley / 2019

"""


import numpy as np
from scipy import interpolate


def interpolate_values_2D(xs, ys, org_data):
    """

    Interpolates given data; defines functions based on this.
    First function gives relative displacement (difference from first 
    time frame); second absolute (relative to origo in given frame).

    Args:
        xs - x coordinates
        ys - y coordinates
        org_data - displacement data; X x Y x 2 numpy array

    Returns:
        function fn_abs : R2 - R2 - relative displacement
        function fn_rel : R2 - R2 - absolute displacement

    """

    # Displacement data in x, y directions

    Xs = org_data[:, :, 0].transpose()
    Ys = org_data[:, :, 1].transpose()

    # X-motion / Y-motion : finds the values of displacement of a given
    # point (x,y) on the grid defines by Xs / Ys which is the 2D array
    # that contains all the motion data from the points

    fn_x = interpolate.interp2d(xs, ys, Xs, kind='cubic')
    fn_y = interpolate.interp2d(xs, ys, Ys, kind='cubic')
    
    fn_rel = lambda x, y: np.array([float(fn_x(x, y)), float(fn_y(x, y))])
    fn_abs = lambda x, y: np.array([x, y]) - fn_rel(x, y)

    return fn_abs, fn_rel

