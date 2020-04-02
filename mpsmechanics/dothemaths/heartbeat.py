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

    maxima = calc_beat_maxima(data, disp_threshold=disp_threshold)

    if len(maxima) < 3:
        return []

    midpoints = [
        int((maxima[i] + maxima[i + 1]) / 2)
        for i in range(len(maxima) - 1)
    ]

    dist1 = midpoints[1] - midpoints[0]
    if dist1 < midpoints[0]:
        midpoints = [midpoints[0] - dist1] + midpoints

    dist2 = midpoints[-1] - midpoints[-2]
    if midpoints[-1] + dist2 < len(data):
        midpoints += [midpoints[-1] + dist2]

    intervals = [
        (midpoints[i], midpoints[i + 1])
        for i in range(len(midpoints) - 1)
    ]

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

    maxima, _ = find_peaks(
        disp_over_time,
        height=max(disp_over_time) / 2,
        distance=disp_threshold,
    )

    return maxima


def _calc_spatial_max(maxima, intervals, disp_folded):
    """

    Calculates index of maximum displacement within each interval.

    Args:
        maxima - indices for overall peaks, over time
        intervals - indices for interval subdivision
        disp_folded - T x X x Y numpy array

    Returns:
        D x X x Y numpy array, integers giving maximum indices
            within each interval for each spatial point

    """

    assert (
        len(disp_folded.shape) == 3
    ), "Error: Unexpected shape for folded distribution."

    intervals_local = intervals[:]

    if maxima[0] < intervals[0][0]:
        intervals_local.append((0, intervals[0][0]))
    if maxima[-1] > intervals[-1][1]:
        intervals_local.append(
            (intervals[-1][1], disp_folded.shape[0] - 1)
        )

    _, x_dim, y_dim = disp_folded.shape
    argmax_list = np.zeros((len(intervals_local), x_dim, y_dim))

    for (i, (start_in, stop_in)) in enumerate(intervals_local):
        argmax_list[i] = start_in + np.argmax(
            disp_folded[start_in:stop_in], axis=0
        )

    return argmax_list.astype(int)


def calc_beatrate(disp_folded, maxima, intervals, time):
    """

    Estimates beatrate as a metric, per spatial point (x, y).

    Args:
        disp_folded - T x X x Y numpy array (displacement)
        maxima - indices for overall peaks, over time
        intervals - indices for interval subdivision
        time - numpy array giving time steps

    Returns:
        D x X x Y numpy array, where D = num_intervals; each value
            contains difference between two intervals in given
            spatial point
    """

    if len(maxima) < 3:
        return np.nan

    argmax = _calc_spatial_max(maxima, intervals, disp_folded)
    num_intervals = len(argmax)
    _, x_dim, y_dim = disp_folded.shape

    beatrate_spatial = np.zeros((num_intervals - 1, x_dim, y_dim))
    unit_in_ms = 1e3

    for i in range(num_intervals - 1):
        for _x in range(x_dim):
            for _y in range(y_dim):
                start_in = argmax[i, _x, _y]
                stop_in = argmax[i + 1, _x, _y]
                beatrate_spatial[i, _x, _y] = unit_in_ms / (
                    time[stop_in] - time[start_in]
                )

    return beatrate_spatial
