
import numpy as np
from scipy.ndimage import gaussian_filter

from mpsmechanics.dothemaths.operations import calc_norm_over_time

def refine(motion_data, factor, sigma):

    for s in sigma:
        assert s >= 0, f"Error: sigmas {sigma} not all >= 0."

    #refined = np.repeat(np.repeat(np.repeat(motion_data, N_t, axis=0), N_xy, axis=1), N_xy, axis=2)
    
    return gaussian_filter(motion_data, sigma)
