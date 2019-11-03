
import numpy as np
from scipy.ndimage import gaussian_filter


def refine(motion_data, ref_factor, sigma):

    if ref_factor > 1:
        refined = (1/ref_factor)*(np.repeat(np.repeat(motion_data, ref_factor, axis=1), ref_factor, axis=2))
    else:
        refined = motion_data

    return gaussian_filter(refined, sigma)
