"""

Given data on displacement, this script finds average and overall
values for a number of features: Average of each of
    - beat rate
    - displacement
    - x motion
    - prevalence
    - principal strain

Figures for alignment as well as all the characteristic values are
plotted and saved in "Plots"; each is saved both as a png and as a
svg file.

#TODO also make plotting maximum points optional

Åshild Telle / Simula Research Labratory / 2019

"""

import sys
import numpy as np

import io_funs as io
import preprocessing as pp
import operations as op
import angular as an
import heart_beat as hb
import mechanical_properties as mc
import metric_plotting as mp

def add_plt_information(ppl, idt, Tmax):
    """
    Save some general information about the metrics calculated in
    this file. These are used for visual output (plots) for
    filenames, labels, etc.

    Arguments:
        ppl - dictionary which will be altered
        idt - string identity of given data set
        Tmax - time frame for experiment

    """


    suffixes = ["beat_rate", "disp", "xmotion", "ymotion",
            "prevalence", "prstran", "xprstrain", "yprstrain"]

    titles = ["Beat rate", "Displacement", "X-motion",
            "Y-motion", "Prevalence", "Principal strain",
            "Principal strain (X)", "Principal strain (Y)"]

    yscales = [None, (0, 1), (0, 1), (0, 1), (0, 1), None, (0, 1), \
            (0, 1)]

    for i in range(8):
        if i in ppl.keys():
            ppl[i]["idt"] = idt + suffixes[i]
            ppl[i]["title"] = titles[i]
            ppl[i]["yscale"] = yscales[i]
            ppl[i]["Tmax"] = Tmax


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
 
        # TODO plot vertical lines too!

        hb.plot_maxima(disp_t, maxima, plt_pr, plt_id)

    return np.mean(beat)


def _calc_displacement(maxima, disp_t, plt_pr, plt_id, scale): 
    """

    Calculates average of displacement at given maximum indices.

    Arguments:
        maxima - list of indices
        disp_t - displacement over time
        plt_pr - plotting properties dictionary
        plt_id - identity for value of interest, used for plotting

    Returns:
        Average displacement

    """

    disp_scaled = scale*disp_t

    if plt_pr[plt_id]["plot"]:
        hb.plot_maxima(disp_scaled, maxima, plt_pr, plt_id)
    
    return np.mean(np.array([disp_t[m] for m in maxima]))


def _calc_an_projection(maxima, data_xy, e_i, plt_pr, plt_id, scale):
    """

    Calculates average of projectet values at given maximum indices.

    Arguments:
        maxima - list of indices
        data_t - given data over time
        plt_pr - plotting properties dictionary
        plt_id - identity for value of interest, used for plotting

    Returns:
        Average value of projection values

    """

    an_xy = an.calc_projection_vectors(data_xy, e_i, over_time=True)
    an_motion = op.calc_norm_over_time(an_xy)
   
    an_scaled = scale*an_motion

    if plt_pr[plt_id]["plot"]:
        hb.plot_maxima(an_scaled, maxima, plt_pr, plt_id)

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
        hb.plot_maxima(prevalence, maxima, plt_pr, plt_id)

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
        hb.plot_maxima(pr_strain, maxima, plt_pr, plt_id)

    return np.mean(np.array([pr_strain[m] for m in maxima]))
    


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

    T = disp_data.shape[0]
    T_max = dt*T
    threshold = 2*10E-6*dt/scale

    # find some basis data
    
    pr_strain_xy = mc.calc_principal_strain(disp_data, over_time=True)
    disp_data_t = op.calc_norm_over_time(disp_data)

    scale = 1/max(disp_data_t)

    # and some useful variables

    maxima = hb.calc_beat_maxima_time(disp_data_t, scale, T_max, \
            plt_pr)
    e_alpha, e_beta = an.calc_direction_vectors(disp_data, \
            plt_pr)
    
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
            (maxima, disp_data, e_alpha, plt_pr, plt_ids[2], scale),
            (maxima, disp_data, e_beta, plt_pr, plt_ids[3], scale),
            (maxima, disp_data, e_beta, plt_pr, plt_ids[4]),
            (maxima, pr_strain_xy, plt_pr, plt_ids[5]),
            (maxima, pr_strain_xy, e_alpha, plt_pr, plt_ids[6], scale),
            (maxima, pr_strain_xy, e_beta, plt_pr, plt_ids[7], scale)]

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

