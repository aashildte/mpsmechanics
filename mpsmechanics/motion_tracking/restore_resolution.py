
import numpy as np
from scipy.ndimage import median_filter


def refine(motion_data, factor):

    T, X, Y, D = motion_data.shape

    new_data = np.tile(motion_data, (1, factor, factor, 1))

    return median_filter(new_data, size=factor)
