# -*- coding: utf-8 -*-

"""

Calculates beat rate + gives interval splitting based on maxima

Åshild Telle / Simula Research Labratory / 2018-2019

"""

import os
import numpy as np

from . import operations as op

def _get_local_intervals(disp_norm, eps):
    """
    Given displacement over time, this tries to calculate the maxima 
    of each beat.

    The idea is to find local maximum regions by cutting of all
    values below the mean. If the maximum in a local region is below
    (1 + eps)*mean we won't include it (attempting to remove small
    local minima close to the cut-of).

    Args:
        disp_norm - 1D numpy array of dimensions T, displacement over time.
        eps - buffer value, maxima needs to be above (1+eps)*average value

    Returns:
        list of local intervals

    """    

    T = len(disp_norm)

    # find mean and a threshold value

    q1 = np.mean(disp_norm)
    q2 = (1 + eps)*q1

    local_intervals = []

    # iterate through data set

    t, t_start, t_stop = 0, 0, 0

    started = False
    threshold = False

    # skip first interval if we start at a peak
    tt = 0
 
    while disp_norm[tt] > q2:
        tt = tt + 1

    # find intervals

    for t in range(tt, T):

        # crossing from below
        
        if disp_norm[t] > q1:
            if not started :
                t_start = t
                started = True
            elif disp_norm[t] > q2:
                threshold = True
 
        # crossing from above

        if(threshold and disp_norm[t] < q1):
            t_stop = t
            local_intervals.append((t_start, t_stop))
            started = False
            threshold = False

    return local_intervals


def _get_beat_maxima(disp_norm, local_intervals):
    """
    From data on displacement over time, this function calculates
    the indices of the maxima of each beat.

    Args:
        disp_norm - 1D numpy array of dimensions T, disp. over time
        local_intervals - list of intervals containing a maximum point

    Returns:
        numpy array of maxima indices

    """

    maxima = []

    for (x1, x2) in local_intervals:
        m = max(disp_norm[x1:x2])
        m_ind = list(disp_norm[x1:x2]).index(m) + x1
        maxima.append(m_ind)
    
    maxima = np.array(maxima)

    return maxima


def calc_beat_maxima_time(data):
    """

    From data on displacement over time only, this function
    calculates the indices of the maxima of each beat.

    Args:
        data - numpy array, displacement values over time

    Returns:
        list of maxima indices
        
    """
    
    return _get_beat_maxima(data, \
            _get_local_intervals(data, eps=0.05))


def calc_beat_maxima_2D(data):
    """

    From data on displacement over space and time, this function
    calculates the indices of the maxima of each beat.

    Args:
        data   - T x X x Y x 2 numpy array, displacement values

    Returns:
        list of maxima indices
        
    """

    disp_norm = op.calc_norm_over_time(data)

    return calc_beat_maxima_time(disp_norm)
