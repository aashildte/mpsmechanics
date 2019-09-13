# -*- coding: utf-8 -*-
"""

Computes mechanical quantities over space and time.

Åshild Telle / Simula Research Labratory / 2019

"""

import numpy as np
import mps

from ..motion_tracking.motion_tracking import track_motion
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

    xmotion = calc_projection_fraction(displacement, angle)

    mstos = 1E3

    velocity = mstos*np.divide(np.gradient(displacement, axis=0), np.gradient(time)[:,None,None,None]) 

    #threshold = 2       # um/s
    #prevalence = (velocity > threshold).astype(int)
    
    principal_strain = calc_principal_strain(displacement, scale)

    return displacement, xmotion, velocity, principal_strain

    
def _calc_beatrate(disp_folded, maxima, intervals):

    data = {"metrics_max_avg" : {},
            "metrics_avg_avg" : {}}

    _, X, Y, _ = disp_folded.shape
    num_beats = len(maxima) - 1

    beatrate_spatial = np.zeros(num_beats, X, Y, 1)

    i1, i2 = intervals[0]
    #argmax_last_interval = np.argmax(disp_folded[0:i1])

    for i in range(num_beats):
        pass 


    if len(d_all["maxima"]) > 1:
        d_all["metrics_max_avg"]["beatrate"] = 1/dt*np.max(np.diff(d_all["maxima"]))
        d_all["metrics_avg_avg"]["beatrate"] = 1/dt*np.mean(np.diff(d_all["maxima"]))
    else:
        d_all["metrics_max_avg"]["beatrate"] = np.nan
        d_all["metrics_avg_avg"]["beatrate"] = np.nan






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

    displacement, xmotion, velocity, principal_strain = \
            _calc_mechanical_quantities(disp_data, scale, angle, mt_data.time_stamps)

    # over space and time (original data)

    values = {"displacement" : displacement,
              "velocity" : velocity,
              "xmotion" : xmotion,
              "principal strain" : principal_strain}   

    d_all = chip_statistics(values, disp_data, dt) 
    d_all["units"] = {"displacement" : r"$\mu m$",
                      "velocity" : r"$\mu m / s$",
                      "xmotion" : r"-",
                      "principal strain" : r"-"}

    d_all["time"] = mt_data.time_stamps

    print("Done calculating mechanical quantities for " + input_file)

    if save_data:
        save_dictionary(input_file, "analyze_mechanics" , d_all)

    return d_all
