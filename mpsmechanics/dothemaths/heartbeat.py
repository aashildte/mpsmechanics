
"""

Calculates beat rate + gives interval splitting based on maxima

Aashild Telle / Simula Research Labratory / 2018-2019

"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from . import operations as op

def _get_local_intervals(disp_norm, eps):
    """

    Given displacement over time, this tries to calculate the maxima 
    of each beat.

    The idea is to find local maximum regions by cutting of all
    values below the mean. If the maximum in a local region is below
    (1 + eps)*mean we won't include it (attempting to remove small
    local minima close to the cut-of).

    Arguments:
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
    while(disp_norm[tt] > q2):
        tt = tt + 1

    # find intervals

    for t in range(tt, T):

        # crossing from below
        
        if(disp_norm[t] > q1):
            if(not started):
                t_start = t
                started = True
            elif (disp_norm[t] > q2):
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

    Arguments:
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


def calc_beat_maxima_time(data, scale, \
        plt_pr = {"visual check" : False}):
    """

    From data on displacement over time only, this function
    calculates the indices of the maxima of each beat.

    Arguments:
        data - numpy array, displacement values over time
        plt_pr - dictionary defining visual output

    Returns:
        list of maxima indices
        
    """
    
    eps = 0.05
    
    local_intervals = _get_local_intervals(data, eps)
    maxima = _get_beat_maxima(data, local_intervals)
    
    if(plt_pr["visual check"]):
        _plot_disp_thresholds(data, scale, maxima, eps, plt_pr)

    return maxima


def calc_beat_maxima_2D(data, movement, scale=1, Tmax=1, \
        plt_pr = {"visual check" : False}):
    """

    From data on displacement over space and time, this function
    calculates the indices of the maxima of each beat.

    Arguments:
        data   - T x X x Y x 2 numpy array, displacement values
        Tmax  - last time value, optional
        plt_pr - dictionary defining visual output, optional

    Returns:
        list of maxima indices
        
    """

    disp_norm = op.calc_norm_over_time(data, movement)
    
    return calc_beat_maxima_time(disp_norm, scale, plt_pr)


def _plot_disp_thresholds(disp_norm, scale, maxima, eps, plt_pr):
    """

    Plots values, maxima, mean value and mean value*(1+eps) for a 
    visual check.

    Plots are saved as 
        idt + _mean.png
    in a folder called "Figures".

    Arguments:
        disp_norm - displacement over time
        scale     - scale to get in SI units
        maxima    - time steps for maxima values
        eps       - buffer value
        plt_pr    - gives general values

    """
   
    path = plt_pr["path"]
    Tmax = plt_pr["Tmax"]
 
    t = np.linspace(0, Tmax, len(disp_norm))
   
    disp_scaled = scale*disp_norm

    plt.plot(t, disp_scaled)
    
    mean = np.mean(disp_scaled)

    mean_vals = mean*np.ones(len(t))
    mean_eps  = (mean*(1 + eps))*np.ones(len(t))

    plt.plot(t, mean_vals, 'r')
    plt.plot(t, mean_eps, 'g')
    
    m_t = [t[m] for m in maxima]
    max_vals = [disp_scaled[m] for m in maxima]

    plt.scatter(m_t, max_vals, color='red')   

    plt.legend(['Displacement', 'Mean value', \
        'Mean (1 + $\epsilon$)', 'Maxima'], loc=4)

    plt.xlabel('Time (s)')

    filename = os.path.join(path, "mean.png")
    plt.savefig(filename, dpi=1000)

    plt.clf()



if __name__ == "__main__":

    data = np.random.rand(3, 3, 3, 2)
    T = data.shape[0]

    assert(calc_beat_maxima_time(op.calc_norm_over_time(data), 1, T) is not None)
    assert(calc_beat_maxima_2D(data, 1, T) is not None)

    print("All checks passed for heart_beat.py")
