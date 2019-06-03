
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

    '''Xs = org_data[:,:,0].transpose()
    Ys = org_data[:,:,1].transpose()

    fn_x = interpolate.interp2d(xs, ys, Xs, kind='cubic')
    fn_y = interpolate.interp2d(xs, ys, Ys, kind='cubic')
    
    fn1 = lambda x, y: np.array([fn_x(x, y)[0], fn_y(x, y)[0]])
    fn2 = lambda x, y: np.array([x, y]) + fn1(x, y)
   
    return fn1, fn2'''

    Xs = org_data[:, :, 0].transpose()  # contains the X displacement data
    Ys = org_data[:, :, 1].transpose()  # contains the Y displacement data

    fn_x = interpolate.interp2d(xs, ys, Xs,
                                kind='cubic')  # X-motion : finds the values of displacement of a point given (x,y) on
    # the grid defines by Xs which is the 2D array that contains all the motion data from the points
    fn_y = interpolate.interp2d(xs, ys, Ys,
                                kind='cubic')  # Y-Motion : finds the values of displacement of a point given (x,y) on
    # the grid defines by Ys which is the 2D array that contains all the motion data from the points

    fn_rel = lambda x, y: np.array([float(fn_x(x, y)), float(fn_y(x, y))])  # computes the displacement in this frame
    fn_abs = lambda x, y: np.array([x, y]) - fn_rel(x, y)  # computes the new abs position on the point

    return fn_abs, fn_rel

