
import numpy as np
from scipy import interpolate


def interpolate_values_2D(xs, ys, org_data):
    """

    Interpolates given data; defines functions based on this.
    First function gives relative displacement; second absolute.

    Args:
        xs - x coordinates
        ys - y coordinates
        org_data - displacement data; X x Y x 2 numpy array

    Returns:
        function f : R2 - R2 - relative
        function g : R2 - R2 - absolute

    """

    Xs = org_data[:,:,0].transpose()
    Ys = org_data[:,:,1].transpose()

    fn_x = interpolate.interp2d(xs, ys, Xs, kind='cubic')
    fn_y = interpolate.interp2d(xs, ys, Ys, kind='cubic')
    
    fn1 = lambda x, y: np.array([fn_x(x, y)[0], fn_y(x, y)[0]])
    fn2 = lambda x, y: np.array([x, y]) + fn1(x, y)
   
    return fn1, fn2

