"""

Given data on displacement, this script finds average and overall
values for a number of features.

Åshild Telle / Simula Research Labratory / 2019

"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from ..dothemaths import operations as op
from ..dothemaths import angular as an
from ..dothemaths import heartbeat as hb

from . import metric as mt
from . import metric_xy as mt_xy

def calc_metrics(disp_data, ind_list, scale, dt, plt_pr, \
        threshold, movement):
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
        threshold - for prevalence
        movement - movement filter; numpy boolean array

    Returns:
        List of values:
          average beat rate
          average displacement
          average displacement in x direction
          average displacement in y direction
          average velocity
          average velocity in x direction
          average velocity in y direction
          average principal strain
          average principal strain in x direction 
          average principal strain in y direction
          average prevalence

    where the average is taken over all maxima.

    """
    
    # a few parameters

    T, X, Y = disp_data.shape[:3]
    T_max = dt*T
    time = np.linspace(0, T_max, T)

    # and some useful variables - TODO these might be stored in a "cache"?
    disp_time = op.calc_norm_over_time(disp_data, movement)
    maxima = hb.calc_beat_maxima_time(disp_time, scale, plt_pr)
    e_alpha, e_beta = an.calc_direction_vectors(disp_data, plt_pr, movement)
    
    e_alpha = (e_alpha, "x_projection", "X projection")
    e_beta = (e_beta, "y_projection", "Y projection")

    # scale threshold for prevalence

    threshold *= scale

    # check if this is a useful data set or not
    
    if(len(maxima)<=1):
        print("Empty sequence – no intervals found")
        return []

    # calculate and gather relevant information ...

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

    args =    [(),
               (disp_data, None, movement),
               (disp_data, e_alpha, movement),
               (disp_data, e_beta, movement),
               (disp_data, None, movement),
               (disp_data, e_alpha, movement),
               (disp_data, e_beta, movement),
               (disp_data, None, movement),
               (disp_data, e_alpha, movement),
               (disp_data, e_beta, movement),
               (disp_data, threshold, None, movement)]

    values = []
    headers = []
     
    for i in ind_list:
        m = metrics[i](*args[i])

        values.append(m.calc_metric_value(maxima))
        headers.append(m.get_header())

        if(plt_pr[i]['plot']):
            m.plot_metric_time(time, disp_time, maxima, plt_pr["path"])

    return headers, values

