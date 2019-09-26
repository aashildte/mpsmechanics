# -*- coding: utf-8 -*-

"""

Calculates beat rate + gives interval splitting based on maxima

Åshild Telle / Simula Research Laboratory / 2018-2019

"""

import numpy as np
from scipy.signal import find_peaks


def calc_beat_intervals(data, disp_threshold=20):
    """

    From data on displacement over time only, this function
    calculates the interval which defines each beat based on
    given maximum values.

    Args:
        disp_over_time - numpy array, displacement values over time
        dist_threshold - minimum requirement for length of a peak
            to be counted as a peak; default value 20

    Returns:
        list of 2-tuples, defining the beat intervals

    """

    maxima = calc_beat_maxima(data,
                              disp_threshold=disp_threshold)

    if len(maxima) < 3:
        return []

    midpoints = [int((maxima[i] + maxima[i+1])/2) \
            for i in range(len(maxima)-1)]

    dist1 = midpoints[1] - midpoints[0]
    if dist1 < midpoints[0]:
        midpoints = [midpoints[0] - dist1] + midpoints

    dist2 = midpoints[-1] - midpoints[-2]
    if midpoints[-1] + dist2 < len(data):
        midpoints += [midpoints[-1] + dist2]

    intervals = [(midpoints[i], midpoints[i+1]) \
            for i in range(len(midpoints)-1)]

    return intervals


def calc_beat_maxima(disp_over_time, disp_threshold=20):
    """

    From data on displacement over time only, this function
    calculates the indices of the maxima of each beat.

    Args:
        disp_over_time - numpy array, displacement values over time
        dist_threshold - minimum requirement for length of a peak
            to be counted as a peak; default value 20

    Returns:
        list of integers, defining the maximum indices

    """

    maxima, _ = find_peaks(disp_over_time, \
                           height=max(disp_over_time)/2, \
                           distance=disp_threshold)

    return maxima

def _calc_spatial_max(num_intervals, intervals, disp_folded):
    argmax_list = []

    for (start_in, stop_in) in intervals:
        argmax_list += [(start_in + \
                np.argmax(disp_folded[start_in:stop_in], axis=0))]

    return np.array(argmax_list)

def calc_beatrate(disp_folded, maxima, intervals, time):
    """

    Args:

    Returns:

    """

    if len(maxima) < 3:
        return [np.array]*3

    _, x_dim, y_dim = disp_folded.shape

    num_intervals = len(intervals)

    beatrate_spatial = np.zeros((num_intervals-1, x_dim, y_dim))

    argmax_list = _calc_spatial_max(num_intervals, intervals,
                                    disp_folded)

    for i in range(len(argmax_list)-1):
        for _x in range(x_dim):
            for _y in range(y_dim):
                start_in = argmax_list[i, _x, _y]
                stop_in = argmax_list[i+1, _x, _y]
                beatrate_spatial[i, _x, _y] = \
                        1e3 / (time[stop_in] - time[start_in])

    beatrate_avg = np.array([np.mean(beatrate_spatial[i]) \
                for i in range(num_intervals-1)])
    beatrate_std = np.array([np.std(beatrate_spatial[i]) \
                for i in range(num_intervals-1)])

    return beatrate_spatial, beatrate_avg, beatrate_std
