# -*- coding: utf-8 -*-
"""

Computes mechanical quantities over space and time.

Åshild Telle / Simula Research Labratory / 2019

"""

import numpy as np

from ..dothemaths.mechanical_quantities import calc_principal_strain
from ..dothemaths.angular import calc_projection_values
from ..dothemaths.statistics import chip_statistics
from ..utils.iofuns import motion_data as md
from ..utils.iofuns.save_values import save_dictionary


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
        displacement - T x X x Y x 2
        xmotion - T x X x Y x 1
        velocity - T x X x Y x 2
        principal strain - T x X x Y x 2

    """

    displacement = (1/scale)*displacement       # pixels to to um

    t_dim, x_dim, y_dim, _ = displacement.shape
    vector_shape = (t_dim, x_dim*y_dim, 2)
    scalar_shape = (t_dim, x_dim*y_dim, 1)

    xmotion = calc_projection_values(displacement, \
            angle).reshape(scalar_shape)

    velocity = 1/dt*(np.gradient(displacement, \
            axis=0)).reshape(vector_shape)
    
    # TODO speed up this step
    principal_strain = calc_principal_strain(displacement, \
            over_time=True).reshape(vector_shape)
    
    displacement = displacement.reshape(vector_shape)

    return displacement, xmotion, velocity, principal_strain


def analyze_mechanics(input_file, method, save_data=True):
    """

    Args:
        input_file - file name; either nd2 or npy file
        save_data - to store values or not; default value True

    Returns:
        dictionary with relevant output values (TBA)

    """

    # read + preprocess data
    print("input file: ", input_file)

    disp_data, scale, angle, dt, _, _ = md.read_mt_file(input_file, method)

    displacement, xmotion, velocity, principal_strain = \
            _calc_mechanical_quantities(disp_data, scale, angle, dt)

    # over space and time (original data)

    values = {"displacement" : displacement,
              "velocity" : velocity,
              "xmotion" : xmotion,
              "principal strain" : principal_strain}

    d_all = chip_statistics(values, disp_data, dt)

    d_all["units"] = {"displacement" : r"$\mu m$",
                      "velocity" : r"$\mu m / s$",
                      "xmotion" : r"??",
                      "principal strain" : r"(-)"}

    if save_data:
        save_dictionary(input_file, "analyze_mechanics", d_all)

    return d_all
