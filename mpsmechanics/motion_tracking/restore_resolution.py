
import numpy as np
from scipy.ndimage import gaussian_filter

from mpsmechanics.dothemaths.operations import calc_norm_over_time

def apply_filter(motion_data, type_filter, sigma):

    if type_filter=="gaussian":
        return gaussian_filter(motion_data, [0, sigma, sigma, 0])
    elif type_filter=="downsampling":
        sigma = int(sigma)
        T, X, Y, D = motion_data.shape
        X_d = X // sigma
        Y_d = Y // sigma

        square = np.zeros((sigma, sigma))
        
        for t in range(T):
            for x in range(X_d):
                for y in range(Y_d):
                    for d in range(D):
                        avg = np.mean(motion_data[t, \
                                (sigma*x):(sigma*(x+1)), (sigma*y):(sigma*(y+1)), d])

                        for x2 in range(sigma*x, sigma*(x+1)):
                            for y2 in range(sigma*y, sigma*(y+1)):
                                motion_data[t, x2, y2, d] = avg

        return motion_data
