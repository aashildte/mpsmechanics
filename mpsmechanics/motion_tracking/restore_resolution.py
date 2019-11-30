"""

Åshild Telle / Simula Research Laboratory / 2019

"""

import numpy as np
from scipy.ndimage import gaussian_filter


def apply_filter(motion_data, type_filter, sigma):
    """

    Attempts to restore resolution which was lost due to downsampling to pixel
    (integer) values.

    Args:
        motion_data - T x X x Y x 2 numpy array
        type_filter - "gaussian" or "downsampling"
        sigma - argument to filter method, either
          * sigma in x and y direction for gaussian filter
          * number of pixel blocks to combine using downsampling approach

    Return:
        smoothered / downsampled data, similar shape as the original motion_data

    """

    assert type_filter in ("gaussian", "downsampling"), \
            "Error_ Type filter not recognized."

    if type_filter == "gaussian":
        return gaussian_filter(motion_data, [0, sigma, sigma, 0])

    # else: downsamling
    sigma = int(sigma)
    T, X, Y, D = motion_data.shape
    X_d = X // sigma
    Y_d = Y // sigma

    new_data = np.zeros((T, X_d, Y_d, D))

    for t in range(T):
        for x in range(X_d):
            for y in range(Y_d):
                for d in range(D):
                    avg = np.mean(motion_data[t, \
                            (sigma*x):(sigma*(x+1)), (sigma*y):(sigma*(y+1)), d])

                    new_data[t, x, y, d] = avg

    return new_data
