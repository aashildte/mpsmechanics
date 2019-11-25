
import numpy as np
from scipy.ndimage import gaussian_filter

from mpsmechanics.dothemaths.operations import calc_norm_over_time

def apply_filter(motion_data, type_filter, sigma):

    assert type_filter in ("gaussian", "downsampling"), \
            "Error_ Type filter not recognized."

    if type_filter=="gaussian":
        return gaussian_filter(motion_data, [0, sigma/10, sigma/10, 0])
    elif type_filter=="downsampling":
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
