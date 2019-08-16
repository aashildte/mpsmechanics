# -*- coding: utf-8 -*-
"""

Computes mechanical quantities over space and time.

Åshild Telle / Simula Research Labratory / 2019

"""

import numpy as np
import mps

from ..motion_tracking.motion_tracking import track_motion
from ..dothemaths.mechanical_quantities import calc_principal_strain
from ..dothemaths.angular import calc_projection_values
from ..dothemaths.statistics import chip_statistics
from ..utils.iofuns import motion_data as md
from ..utils.iofuns.save_values import save_dictionary
from ..utils.iofuns.data_layer import read_prev_layer


def _calc_mechanical_quantities(displacement, scale, angle, dt):
    """
    
    Derived quantities - reshape to match expected data structure for
    derived layers
    
    Args:
        displacement - displacement data, T x X x Y x 2 numpy array
        scale - scaling factor (pixels to um)
        angle - angle chamber is tilted with
        dt - time step (frame rate)

    Returns:
        displacement in um - T x X x Y x 2
        xmotion - T x X x Y x 1
        velocity - T x X x Y x 2
        principal strain - T x X x Y x 2

    """

    dims = displacement.shape

    displacement = (1/scale)*displacement

    xmotion = calc_projection_values(displacement, angle).\
            reshape(dims[0], dims[1], dims[2], 1)

    velocity = 1/dt*(np.gradient(displacement, axis=0))

    threshold = 2       # um/s
    prevalence = (velocity > threshold).astype(int)

    principal_strain = calc_principal_strain(displacement)

    return displacement, xmotion, velocity, prevalence, principal_strain


def analyze_mechanics(input_file, save_data=True):
    """

    Args:
        input_file - file name; either nd2 or npy file
        save_data - to store values or not; default value True

    Returns:
        dictionary with relevant output values (TBA)

    """

    mt_data = mps.MPS(input_file)
    data = read_prev_layer(input_file, "track_motion", track_motion, save_data)
   
    disp_data = data["displacement vectors"]
    angle = data["angle"]

    scale = mt_data.info["um_per_pixel"]
    dt = mt_data.dt
    
    displacement, xmotion, velocity, prevalence, principal_strain = \
            _calc_mechanical_quantities(disp_data, scale, angle, dt)
    
    # over space and time (original data)

    values = {"displacement" : displacement,
              "velocity" : velocity,
              "xmotion" : xmotion,
              "principal strain" : principal_strain,
              "prevalence" : prevalence}
    
    d_all = chip_statistics(values, disp_data, dt) 
    d_all["units"] = {"displacement" : r"$\mu m$",
                      "velocity" : r"$\mu m / s$",
                      "xmotion" : r"??",
                      "principal strain" : r"-",
                      "prevalence" : r"-"}
    if save_data:
        save_dictionary(input_file, "analyze_mechanics" , d_all)

    return d_all
