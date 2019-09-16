# -*- coding: utf-8 -*-
"""

Computes mechanical quantities over space and time.

Åshild Telle / Simula Research Labratory / 2019

"""

import numpy as np
import mps
from collections import defaultdict

from ..motion_tracking.motion_tracking import track_motion
from ..motion_tracking.ref_frame import convert_disp_data, calculate_minmax
from ..dothemaths.mechanical_quantities import calc_principal_strain
from ..dothemaths.angular import calc_angle_diff, calc_projection_fraction
from ..dothemaths.statistics import chip_statistics
from ..utils.iofuns import motion_data as md
from ..utils.iofuns.save_values import save_dictionary
from ..utils.iofuns.data_layer import read_prev_layer


def _calc_mechanical_quantities(displacement, scale, angle, time):
    """
    
    Derived quantities - reshape to match expected data structure for
    derived layers
    
    Args:
        displacement - displacement data, T x X x Y x 2 numpy array
        scale - scaling factor (pixels to um)
        angle - angle chamber is tilted with
        time - all time steps

    Returns:
        displacement in um - T x X x Y x 2
        xmotion - T x X x Y x 1
        velocity - T x X x Y x 2
        principal strain - T x X x Y x 2

    """

    displacement = scale*displacement

    displacement_minmax = convert_disp_data(displacement, calculate_minmax(displacement))

    xmotion = calc_projection_fraction(displacement, angle)

    ms_to_s = 1E3

    velocity = ms_to_s*np.divide(np.gradient(displacement, axis=0), \
            np.gradient(time)[:,None,None,None]) 

    principal_strain = calc_principal_strain(displacement, scale)

    return displacement, displacement_minmax, xmotion, velocity, principal_strain

    
def _calc_beatrate(disp_folded, maxima, intervals, time):

    data = defaultdict(dict)

    if len(maxima)<2:
        data["metrics_max_avg"] = np.nan
        data["metrics_avg_avg"] = np.nan
        data["metrics_max_std"] = np.nan
        data["metrics_avg_std"] = np.nan
        return data

    _, X, Y = disp_folded.shape
    num_intervals = len(maxima) - 1

    beatrate_spatial = np.zeros((num_intervals, X, Y))
    beatrate_avg = np.zeros(num_intervals)
    beatrate_std = np.zeros(num_intervals)

    i1 = intervals[0][0]
    argmax_prev = np.argmax(disp_folded[0:i1], axis=0)

    intervals_ext = intervals + [(intervals[-1][1], -1)]

    for i in range(num_intervals):
        i1, i2 = intervals_ext[i] 
        argmax_current = i1 + np.argmax(disp_folded[i1:i2], axis=0)
        
        for x in range(X):
            for y in range(Y):
                j1, j2 = argmax_prev[x, y], argmax_current[x, y]
                beatrate_spatial[i,x,y] = 1E3/(time[j2] - time[j1])
        
        beatrate_avg[i] = np.mean(beatrate_spatial[i])
        beatrate_std[i] = np.std(beatrate_spatial[i])
        np.copyto(argmax_prev, argmax_current)

    data["metrics_max_avg"] = np.max(beatrate_avg) 
    data["metrics_avg_avg"] = np.mean(beatrate_avg)
    data["metrics_max_std"] = np.max(beatrate_std) 
    data["metrics_avg_std"] = np.mean(beatrate_std)

    return beatrate_spatial, beatrate_avg, beatrate_std, data


def analyze_mechanics(input_file, save_data=True):
    """

    Args:
        input_file - file name; either nd2 or npy file
        save_data - to store values or not; default value True

    Returns:
        dictionary with relevant output values (TBA)

    """
    
    mt_data = mps.MPS(input_file)
    data = read_prev_layer(input_file, "track_motion", track_motion, \
            save_data=save_data)

    disp_data = data["displacement vectors"]
    angle = data["angle"]
    
    scale = 8*mt_data.info["um_per_pixel"]
    dt = mt_data.dt 
    
    print("Calculating mechanical quantities for " + input_file)

    displacement, displacement_minmax, xmotion, velocity, principal_strain = \
            _calc_mechanical_quantities(disp_data, scale, angle, mt_data.time_stamps)

    # over space and time (original data)

    values = {"displacement" : displacement,
              "displacement max diff." : displacement_minmax,
              "velocity" : velocity,
              "xmotion" : xmotion,
              "principal strain" : principal_strain}   

    d_all = chip_statistics(values, disp_data, dt) 

    d_all["time"] = mt_data.time_stamps

    br_spa, beatrate_avg, beatrate_std, data_beatrate = \
            _calc_beatrate(d_all["folded"]["displacement"], \
                           d_all["maxima"], d_all["intervals"],
                           d_all["time"])

    d_all["beatrate_spatial"] = br_spa
    d_all["beatrate_avg"] = beatrate_avg
    d_all["beatrate_std"] = beatrate_std

    for k in ["metrics_max_avg", "metrics_avg_avg", \
            "metrics_max_std", "metrics_avg_std"]:
        d_all[k]["beatrate"] = data_beatrate[k]

    d_all["units"] = {"displacement" : r"$\mu m$",
                      "displacement max diff." : r"$\mu m$",
                      "velocity" : r"$\mu m / s$",
                      "xmotion" : r"-",
                      "principal strain" : r"-",
                      "beatrate" : "beats/s"}


    print("Done calculating mechanical quantities for " + input_file)

    if save_data:
        save_dictionary(input_file, "analyze_mechanics" , d_all)

    return d_all
