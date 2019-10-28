
import numpy as np
from scipy.ndimage import median_filter


def refine(motion_data, ref_factor, filter_size):

    if ref_factor > 1:
        refined = np.repeat(np.repeat(motion_data, ref_factor, axis=1), \
                         ref_factor, axis=2)
    else:
        refined = motion_data

    return median_filter(refined, filter_size)
