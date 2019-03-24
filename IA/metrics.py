"""

Given data on displacement, this script finds average and overall
values for a number of features: Average of each of
    - beat rate
    - displacement
    - x motion
    - prevalence
    - principal strain

Figures for alignment as well as all the characteristic values are
plotted and saved in "Plots"; each is saved as a png file.

Åshild Telle / Simula Research Labratory / 2019

"""

import sys
import numpy as np
import matplotlib.pyplot as plt

import io_funs as io
import preprocessing as pp
import operations as op
import angular as an
import heart_beat as hb
import mechanical_properties as mc
import metric_plotting as mp


def _calc_beat_rate(maxima, disp_t, plt_pr, plt_id):
    """

    Calculates beat rate based on displacement: Average of difference
    between given maximum indices.

    Arguments:
        maxima - list of indices
        disp_t - displacement over time, used for plotting
        plt_pr - plotting properties dictionary
        plt_id - identity for value of interest, used for plotting

    Returns:
        Average beat rate

    """
    
    beat = np.array([(maxima[k] - maxima[k-1]) \
                        for k in range(1, len(maxima))])

    if(plt_pr[plt_id]["plot"]):
        plot_metric_values(disp_t, maxima, plt_pr, plt_id, \
                mark_maxima="l")

    return np.mean(beat)


def _calc_displacement(maxima, disp_t, plt_pr, plt_id, scale): 
    """

    Calculates average of displacement at given maximum indices.

    Arguments:
        maxima - list of indices
        disp_t - displacement over time
        plt_pr - plotting properties dictionary
        plt_id - identity for value of interest, used for plotting
        scale - TODO do we need this here?

    Returns:
        Average displacement

    """

    disp_scaled = scale*disp_t

    if plt_pr[plt_id]["plot"]:
        plot_metric_values(disp_scaled, maxima, plt_pr, plt_id)
    
    return np.mean(np.array([disp_t[m] for m in maxima]))


def _calc_an_projection(maxima, data_xy, e_i, plt_pr, plt_id):
    """

    Calculates average of projectet values at given maximum indices.

    Arguments:
        maxima - list of indices
        data_t - given data over time
        plt_pr - plotting properties dictionary
        plt_id - identity for value of interest, used for plotting

    Returns:
        Average value of projection values

    TODO: Consider if the projected values should be scaled somehow
          + should x and y projections be plotted together?

    """

    an_xy = an.calc_projection_vectors(data_xy, e_i, over_time=True)
    an_motion = op.calc_norm_over_time(an_xy)
   
    if plt_pr[plt_id]["plot"]:
        plot_metric_values(an_motion, maxima, plt_pr, plt_id)

    return np.mean(np.array([an_motion[m] for m in maxima]))


def _calc_prevalence(maxima, disp_data, threshold, plt_pr, plt_id):
    """
 
    Calculates prevalence over all time steps.

    Arguments:
        disp_data - displacement, numpy array of dimensions
            T x X x Y x 2
        threshold - should be scaled to unit scales 
        plt_pr - plotting properties dictionary
        plt_id - identity for value of interest, used for plotting

    Returns:
        Average prevalence

    """

    T, X, Y = disp_data.shape[:3]
    scale = 1./(X*Y)              # no of points

    prev_xy = mc.calc_prevalence(disp_data, threshold)
    
    prevalence = np.zeros(T-1)

    for t in range(T-1):
        prevalence[t] = scale*np.sum(prev_xy[t])

    if plt_pr[plt_id]["plot"]:
        plot_metric_values(prevalence, maxima, plt_pr, plt_id)

    return np.mean(np.array([prevalence[m] for m in maxima]))
    

def _calc_principal_strain(maxima, pr_strain_xy, plt_pr, plt_id):
    """

    Calculates average of principal strain at given maximum indices.

    Arguments:
        maxima - list of indices
        data_t - given data over time
        plt_pr - plotting properties dictionary
        plt_id - identity for value of interest, used for plotting

    Returns:
        Average value of principal strain.

    """
    pr_strain = op.calc_norm_over_time(pr_strain_xy)

    if plt_pr[plt_id]["plot"]:
        plot_metric_values(pr_strain, maxima, plt_pr, plt_id)

    return np.mean(np.array([pr_strain[m] for m in maxima]))
    

def plot_metric_values(values, maxima, plt_pr, plt_id, mark_maxima=""):
    """

    Plots values and maxima for visual output.

    Plots are saved as [idt]_[suffix].png in a folder named Figures

    Arguments:
        values      - data over time
        maxima      - list of maxima indices
        plt_pr      - dictionary giving plotting properties
        plt_id      - id for extracting dictionary values
        mark_maxima - string, set to "l" to get vertical lines;
            "o" to get point at maxima or "lo" / "ol" to get both

    """
 
    description = plt_pr[plt_id]["title"]
    T_max = plt_pr[plt_id]["Tmax"]
    y_interval = plt_pr[plt_id]["yscale"]
    idt = plt_pr[plt_id]["idt"]

    t = np.linspace(0, T_max, len(values))
    
    plt.plot(t, values)
    
    # maxima
    m_t = [t[m] for m in maxima]
    max_vals = [values[m] for m in maxima]
    
    if("l" in mark_maxima):
        for t in m_t:
            plt.axvline(x=t, color='red')
    if("o" in mark_maxima):
        plt.scatter(m_t, max_vals, color='red')

    # visual properties
    plt.xlabel('Time (s)')
    plt.title(description)

    if y_interval is not None:
        plt.ylim(y_interval[0], y_interval[1])
    
    # save as ...
    de = io.get_os_delimiter()

    plt.savefig(plt_pr["path"] + de + idt + ".png", dpi=1000)

    plt.clf()


def get_numbers_of_interest(disp_data, ind_list, scale, dt, plt_pr):
    """
    
    Threshold given by 2 um/s; emperically, from paper. We scale to
    get on same units as displacement data, and to get dt on a unit
    scale in prevalence test.

    Arguments:
        disp_data - displacement data, scaled
        ind_list - list of integers, indicating which values to
            calculate
        scale - convert disp_data to original values to get
            the original magnitude
        dt - temporal difference
        dx - spacial difference
        plt_pr - dictionary determining visual output

    Returns:
        List of values:
          average beat rate
          average displacement
          average displacement in x direction
          average displacement in y direction
          average prevalence
          average principal strain
          average principal strain in x direction 
          average principal strain in y direction

    where each value is taken over peak values only.

    """
    
    # a few parameters

    T, X, Y = disp_data.shape[:3]
    T_max = dt*T
    threshold = 2*10E-6*dt/scale
    scale_overall = 1/(X*Y)

    # find some basis data
    
    pr_strain_xy = scale_overall*mc.calc_principal_strain(disp_data, \
            over_time=True)
    disp_data_t = scale_overall*op.calc_norm_over_time(disp_data)

    # and some useful variables

    maxima = hb.calc_beat_maxima_time(disp_data_t, scale, T_max, \
            plt_pr)
    e_alpha, e_beta = an.calc_direction_vectors(disp_data, plt_pr)
    
    # check if this is a useful data set or not
    
    if(len(maxima)<=1):
        print("Empty sequence – no intervals found")
        return []

    #plt_ids = range(8)
    plt_ids = [mp.get_pr_id(x) for x in \
            ["beat_rate", "displacement", "xmotion", "ymotion",
            "prevalence", "prstrain", "xprstrain", "yprstrain"]]

    # calculate and gather relevant information ...

    fns = [_calc_beat_rate, _calc_displacement, _calc_an_projection,
            _calc_an_projection, _calc_prevalence,
            _calc_principal_strain, _calc_an_projection,
            _calc_an_projection]

    args = [(maxima, disp_data_t, plt_pr, plt_ids[0]),
            (maxima, disp_data_t, plt_pr, plt_ids[1], scale),
            (maxima, disp_data, e_alpha, plt_pr, plt_ids[2]),
            (maxima, disp_data, e_beta, plt_pr, plt_ids[3]),
            (maxima, disp_data, e_beta, plt_pr, plt_ids[4]),
            (maxima, pr_strain_xy, plt_pr, plt_ids[5]),
            (maxima, pr_strain_xy, e_alpha, plt_pr, plt_ids[6]),
            (maxima, pr_strain_xy, e_beta, plt_pr, plt_ids[7])]

    values = []

    for i in ind_list:
        v = fns[i](*args[i])
        values.append(v)

    return values


if __name__ == "__main__":

    try:
        f_in = sys.argv[1]
    except:
        print("Error reading file. Give file name as first argument.")
        exit(-1)
    
    data, scale = io.read_disp_file(f_in, 1)
    T = data.shape[0]

    ppl = {}

    for i in range(8):
        ppl[int(i)] = {"plot" : False}

    ppl["Tmax"] = 1
    ppl["dims"] = (6, 4)
    ppl["visual check"] = False

    indices = range(8)

    assert(get_numbers_of_interest(data, indices, scale, 1, ppl) \
            is not None)
    print("All checks passed for metrics.py")

