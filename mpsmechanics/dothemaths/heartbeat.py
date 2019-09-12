# -*- coding: utf-8 -*-

"""

Calculates beat rate + gives interval splitting based on maxima

Åshild Telle / Simula Research Laboratory / 2018-2019

"""

from scipy.signal import find_peaks


def calc_beat_intervals(data, disp_threshold=30):
    """

    From data on displacement over time only, this function
    calculates the interval which defines each beat based on
    given maximum values.

    Args:
        disp_over_time - numpy array, displacement values over time
        dist_threshold - minimum requirement for length of a peak
            to be counted as a peak; default value 10

    Returns:
        list of 2-tuples, defining the beat intervals
 
    """

    maxima = calc_beat_maxima(data,
                              disp_threshold=disp_threshold)
    midpoints = [int((maxima[i] + maxima[i+1])/2) \
            for i in range(len(maxima)-1)]
    intervals = [(midpoints[i], midpoints[i+1]) \
            for i in range(len(midpoints)-1)]

    return intervals


def calc_beat_maxima(disp_over_time, disp_threshold=30):
    """

    From data on displacement over time only, this function
    calculates the indices of the maxima of each beat.

    Args:
        disp_over_time - numpy array, displacement values over time
        dist_threshold - minimum requirement for length of a peak
            to be counted as a peak; default value 10

    Returns:
        list of integers, defining the maximum indices
        
    """

    maxima, _ = find_peaks(disp_over_time, \
                           height=max(disp_over_time)/2, \
                           distance=disp_threshold)

    return maxima
