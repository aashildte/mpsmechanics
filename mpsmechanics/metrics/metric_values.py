"""

Given data on displacement, this script finds average and overall
values for a number of features.

id mapping:
        0 - beat rate
        1 - displacement
        2 - displacement in x direction
        3 - displacement in y direction
        4 - velocity
        5 - velocity in x direction
        6 - velocity in y direction
        7 - principal strain
        8 - principal strain in x direction 
        9 - principal strain in y direction
        10 - prevalence

Åshild Telle / Simula Research Labratory / 2019

"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from ..dothemaths import preprocessing as pp
from ..dothemaths import operations as op
from ..dothemaths import angular as an
from ..dothemaths import heartbeat as hb

from . import metric as mt
from . import metric_xy as mt_xy


def _metric_setup(disp_data, scale, dt, threshold_pr, threshold_mv, plt_pr):

    # some useful variables - TODO these might be stored in a "cache"?
    movement = pp.calc_filter(disp_data, threshold_mv)
    disp_time = op.calc_norm_over_time(disp_data, movement)
    maxima = hb.calc_beat_maxima_time(disp_time, scale, plt_pr)

    e_alpha, e_beta = an.calc_direction_vectors(disp_data, plt_pr, movement)
    e_alpha = (e_alpha, "x_projection", "X projection")
    e_beta = (e_beta, "y_projection", "Y projection")

    threshold_pr /= scale/dt

    metrics = [mt.Beatrate,
               mt_xy.Displacement,
               mt_xy.Displacement,
               mt_xy.Displacement,
               mt_xy.Velocity,
               mt_xy.Velocity,
               mt_xy.Velocity,
               mt_xy.Principal_strain,
               mt_xy.Principal_strain,
               mt_xy.Principal_strain,
               mt_xy.Prevalence]



    args =    [(disp_time, maxima),
               (disp_data, None, movement, maxima),
               (disp_data, e_alpha, movement, maxima),
               (disp_data, e_beta, movement, maxima),
               (disp_data, None, movement, maxima),
               (disp_data, e_alpha, movement, maxima),
               (disp_data, e_beta, movement, maxima),
               (disp_data, None, movement, maxima),
               (disp_data, e_alpha, movement, maxima),
               (disp_data, e_beta, movement, maxima),
               (disp_data, threshold_pr, None, movement, maxima)]

    return metrics, args, maxima


def plot_metrics2D(disp_data, dimensions, ind_list, scale, dt,
        threshold_pr, threshold_mv, path, over_time):

    plt_pr = {"visual check" : False}

    metrics, args, _ = _metric_setup(disp_data, scale, dt, \
            threshold_pr, threshold_mv, plt_pr)

    values = []
    headers = []
     
    for i in ind_list:
        m = metrics[i](*args[i])
        m.plot_spacial_dist(dimensions, path, over_time)


def calc_metrics(disp_data, ind_list, scale, dt, plt_pr, \
        threshold_pr, threshold_mv):
    """

    Arguments:
        disp_data - displacement data, scaled
        ind_list - list of integers, indicating which values to
            calculate
        scale - convert disp_data to original values to get
            the original magnitude
        dt - temporal difference
        dx - spacial difference
        plt_pr - dictionary determining visual output
        threshold_pr - for prevalence
        threshold_mv - for movement

    Returns:
        List of metrics, based on average values

    """

    # disp data:

    T, X, Y, d = disp_data.shape

    for t in range(len(disp_data)):

        A = 0

        for x in range(X):
            for y in range(Y):
                A += np.linalg.norm(disp_data[t, x, y])
        
        print(A/(X*Y))
    # a few parameters

    T, X, Y = disp_data.shape[:3]
    T_max = dt*T
    time = np.linspace(0, T_max, T)

    # calculate and gather relevant information ...

    metrics, args, maxima = _metric_setup(disp_data, scale, dt, \
            threshold_pr, threshold_mv, plt_pr)
    
    # check if this is a useful data set or not
    
    if(len(maxima)<=1):
        print("Empty sequence – no intervals found")
        return [], []
    
    values = []
    headers = []
     
    for i in ind_list:
        m = metrics[i](*args[i])

        values.append(m.calc_metric_value())     # TODO -> average value?
        headers.append(m.get_header())

        if(plt_pr[i]['plot']):
            m.plot_metric_time(time, plt_pr["path"])

    return headers, values
