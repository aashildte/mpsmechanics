"""

Functions for changing reference frame for displacement, assuming we
can calculate the reference frame *after* the blocktracking algorithm
(transitive property).

Åshlid Telle / Simula Research Laboratory / 2019

"""

import numpy as np
import matplotlib.pyplot as plt

from ..dothemaths.operations import calc_norm_over_time

def convert_disp_data(frames, ref_index):
    """

    Converts displacement data in frames to new index.

    Args:
        frames - numpy array of dimensions T x X x Y x D
        ref_index - which index to use as reference

    Returns:
        numpy array of dimensions T x X x Y x D

    """
    assert ref_index >= 0 and ref_index <= frames.shape[0], \
            "Invalid reference index."

    return frames - frames[ref_index]

def _find_longest_subinterval(diff_norm):
    """

    Finds the longest subinterval with values below a predefined
    threshold, currently set to be half of the maximum value.

    Args:
        diff_norm - numpy 1D array, of dimension T

    Returns:
        interval - tuple (x, y), with 0 <= x < y <= T

    """

    threshold = max(diff_norm)/2       # because why not

    indices = list(filter(lambda i: diff_norm[i] < threshold, \
            range(len(diff_norm))))

    print("threshold: ", threshold)

    count = 0
    prev = indices[0]
    start_ind = 0
    max_count = 0

    for i in indices[1:]:
        if i == (prev + 1):
            count += 1
        else:
            if count > max_count:
                max_count = count
                start_ind = i-count
            count = 0
        prev = i

    return (start_ind, start_ind + max_count)


def calculate_min_velocity_frame(frames):
    """

    Hypothesis: Resting state is where velocity is smallest, in a
    longer time frame.

    This function finds a frame which
        1) is among the frames with lowest velocity recorded
        2) is in the longest interval among those found for (1)
        3) is the minimum in this interval

    Derivative represented by smallest difference - assuming that
    the time step is fixed, and that minimum velocity is where
        (d[i] - d[i-1])/dt
    is smallest; for fixed dt this equals where d[i] - d[i-1] is
    smallest.

    Args:
        frames - numpy array of dimensions X x Y x 2 x T

    Returns:
        index for lowest velocity within the longest interval

    """

    diff_norm = np.sum(np.linalg.norm(np.diff(frames), axis=2), \
            axis=(0, 1))
    interval = _find_longest_subinterval(diff_norm)
    
    norm = np.sum(np.linalg.norm(frames, axis=2), \
            axis=(0, 1))

    return interval[0] + np.argmin(diff_norm[interval[0]:interval[1]])


def calculate_firstframe(frames):
    """
    
    Just the first one. For comparison - this SHOULD give the
    same values as the original script!
    
    """
    return 0


def calculate_minmax(frames):
    """

    Hypothesis: The maximum displacement will either actually be a
    maximum or a minimum in the "true" displacement trace. It can be
    determined from the mean which is the case.

    Args:
        frames - numpy array of dimensions T x X x Y x 2

    Returns:
        index for assumed minimum

    """

    norm = calc_norm_over_time(frames)
     
    min_index = np.argmax(norm)
    min_norm = norm[min_index]
    mean_norm = np.mean(norm)
    if (min_norm - mean_norm) > mean_norm:
        frames_shifted = convert_disp_data(frames, min_index)
        norm = calc_norm_over_time(frames_shifted)
        min_index = np.argmax(norm)
    
    return min_index
