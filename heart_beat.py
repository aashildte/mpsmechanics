"""

    Calculates beat rate + gives interval splitting based on maxima

    Aashild Telle / Simula Research Labratory / 2019

"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import read_data as io
import preprocess_data as pp


def _get_local_intervals(disp_norm, eps=.25):
    """

    Given displacement over time, this tries to calculate the maxima 
    of each beat.

    The idea is to find local maximum regions by cutting of all
    values below the mean. If the maximum in a local region is below
    (1 + eps)*mean we won't include it (attempting to remove small
    local minima close to the cut-of).

    Arguments:
        disp_norm - 1D numpy array of dimensions T, displacement over time.
        eps - cut-of value, default .25

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

    for t in range(T):
        if(disp_norm[t] > q1):
            if(not started):
                t_start = t
                started = True
            elif(disp_norm[t] > q2):
                threshold = True
            
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
        list of maxima indices
    """

    maxima = []

    for (x1, x2) in local_intervals:
        m = max(disp_norm[x1:x2])
        m_ind = list(disp_norm[x1:x2]).index(m) + x1
        maxima.append(m_ind)
    
    return maxima


def get_beat_maxima(data, idt, T_max):
    """
    From data on displacement over time, this function calculates
    the indices of the maxima of each beat.

    Arguments:
        data - T x X x Y x 2 numpy array, displacement values
        idt - idt for visulization check plots
        T_max     - last time value

    Returns:
        list of maxima indices
        
    """

    disp_norm = pp.get_overall_movement(data)
    local_intervals = _get_local_intervals(disp_norm)
    maxima = _get_beat_maxima(disp_norm, local_intervals)
    _plot_maxima(disp_norm, maxima, idt, T_max)

    return maxima

def get_average(pts):
    """

    Calculates average difference of a set of points.

    Arguments:
        pts - list of points

    Returns:
        averaged difference

    """

    N = len(pts)
    s = 0

    for k in range(N-1):
        s = s + (pts[k+1] - pts[k])

    return s/(N-1)
         


def _plot_maxima(disp_norm, maxima, idt, T_max):
    """

    Plots values and maxima for visual check.

    Plots are saved as Plots/[idt]_maxima.png and Plots/[idt]_maxima.svg.

    Arguments:
        disp_norm - displacement over time
        maxima    - list of maxima indices
        idt       - filename idt
        T_max     - last time value
    """
    
    # make plot dir if it doesn't already exist
    
    if not (os.path.exists("Plots")):
         os.mkdir("Plots")

    t = np.linspace(0, T_max, len(disp_norm))
    
    m_t = [t[m] for m in maxima]

    plt.plot(t, disp_norm)
    max_vals = [disp_norm[m] for m in maxima]
    plt.scatter(m_t, max_vals, color='red')
    
    plt.savefig("Plots/" + idt + "_maxima.png")
    plt.savefig("Plots/" + idt + "_maxima.svg")

    plt.clf()

if __name__ == "__main__":

    try:
        f_in = sys.argv[1]
        f_ou = sys.argv[2]
    except:
        print("Error reading file. Give file name as first argument, identity for plotting as second.")
        exit(-1)
    
    data = io.read_disp_file(f_in)

    T, X, Y = data.shape[:3]

    maxima = get_beat_maxima(data, "test", T)
    assert(maxima != None)
    print("Beat maxima check passed")

    assert(get_average(maxima) != None)
    print("Average check passed")

    print("All checks passed")
