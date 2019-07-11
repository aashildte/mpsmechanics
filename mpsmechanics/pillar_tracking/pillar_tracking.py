# -*- coding: utf-8 -*-
"""

Point tracking related functions

Åshild Telle / Simula Research Labratory / 2019

"""

import os
import numpy as np

from ..dothemaths.interpolation import interpolate_values_2D
from ..dothemaths.statistics import chip_statistics

from ..utils.iofuns.motion_data import read_mt_file
from ..utils.iofuns.folder_structure import get_input_properties 
from ..utils.iofuns.position_data import read_pt_file
from ..utils.iofuns.save_values import save_dictionary

from .forcetransformation import displacement_to_force, \
        displacement_to_force_area

def _define_pillars(p_values, no_meshpts=200):
    """

    Defines circle to mark the pillar's circumference, based on
    information about the pillar's initial middle position.

    Args:
        p_values - list like structure of dimensions no_pillars x 3
        no_meshpts - number of mesh points used

    Returns:
        p_values - mesh points on the circumsphere of the circle
            defined by x, y, z
    """

    assert len(p_values) > 0, "p_values can't be an empty list"
    assert no_meshpts > 0, "no_meshpts must be greater than 0"

    no_pillars = len(p_values)
    pillars = np.zeros((no_pillars, no_meshpts, 2))
    angles = np.linspace(0, 2*np.pi, no_meshpts)

    for i in range(no_pillars):

        # mesh points on the circumsphere of the circle
        # @David keep the original as default maybe, and
        # try variations in separate branches

        x_pos, y_pos, radius = p_values[i]

        #TODO something like pillars[i, :, 0] = x_pos + radius*np.cos(angles)

        for j in range(no_meshpts):
            pillars[i, j, 0] = x_pos + radius*np.cos(angles[j])
            pillars[i, j, 1] = y_pos + radius*np.sin(angles[j])

    return pillars


def _calculate_current_timestep(x_coords, y_coords, data_disp, pillars):
    """

    Calculates values at given tracking points (defined by pillars)
    based on interpolation.

    Args:
        x_coords - x coordinates, dimension X
        y_coords - y coordinates, dimension Y
        data_disp[t] - numpy array of dimensions X x Y x 2
        pillars - numpy array of dimensions no_pillars x no_meshpts x 2

    Returns:
        numpy array of dimension no_meshpts x 2, middle point (average
            of all points on the circumfence); relative displacement

    """

    fn_rel = interpolate_values_2D(x_coords, y_coords, data_disp)

    no_pillars, no_meshpts, no_dims = pillars.shape

    midpt_values = np.zeros((no_pillars, no_dims))

    # midpoints, relative displacement

    for p in range(no_pillars):
        mean = 0
        for n in range(no_meshpts):
            mean += fn_rel(*pillars[p, n])
        mean /= no_meshpts
        midpt_values[p] = mean

    return midpt_values


def _track_pillars_over_time(data_disp, pillars_mpoints, size_x, size_y):
    """

    Tracks position of mesh poinds defined for pillars over time, based
    on the displacement data_disp.

    Args:
        data_disp - displacement data, numpy array of size T x X x Y x 2
        pillars_mpoints - coordinates + radii of each pillar
        size_x - length (in pixels) of picture
        size_y - width (in pixels) of picture

    Returns:
        numpy array of dimensions T x no_meshpts x no_pillars (no_meshpts default value);
            gives absolute displacement of circumfence points

    """

    # define pillars by their circumference

    pillars = _define_pillars(pillars_mpoints)

    # some general values
    T, x_dim, y_dim = data_disp.shape[:3]
    no_pillars, no_meshpts = pillars.shape[:2]

    x_coords = np.linspace(0, size_x, x_dim)
    y_coords = np.linspace(0, size_y, y_dim)

    # store data - as an average of all meshpoints

    all_values = np.zeros((T, no_pillars, 2))

    for t in range(T):
        all_values[t] = \
                _calculate_current_timestep(x_coords, y_coords, \
                                            data_disp[t], pillars)

    return all_values

def _find_pillar_positions_file(f_disp):
    """

    TODO change to npy files

    Args:
        f_disp - displacement data file

    Returns:
        csv file for pillar positions, expected to be in a subfolder
        with the same name

    """

    path, filename, _ = get_input_properties(f_disp)
    
    return os.path.join(path, filename + "_pillars.csv")


def track_pillars(f_disp, L=50E-6, R=10E-6, E=2.63E-6, \
        save_data=True):
    """

    Tracks points corresponding to "pillars" over time.

    Arguments:
        f_disp - filename for displacement data
        L - ??
        R - ??
        E - ??
        save_data - to store values or not; default value True

    Returns:
        dictionary with calculated values

    """

    assert (".csv" in f_disp) or (".nd2" in f_disp), \
        "Displacement file must be a csv or nd2 file"

    f_pts = _find_pillar_positions_file(f_disp)
    print(f_pts)

    assert os.path.isfile(f_pts), "Pillar position file not found."

    # displacement data and positions of pillars

    data_disp, scaling_factor, angle, dt, size_x, size_y = read_mt_file(f_disp)
    pillars_mpoints = read_pt_file(f_pts, scaling_factor)

    print("Tracking pillars for data set: ", f_disp)

    mdpt_values_px = _track_pillars_over_time(data_disp, \
            pillars_mpoints, size_x, size_y)
    
    # then do a couple of transformations ..

    mdpt_values_um = 1/scaling_factor*mdpt_values_px

    area = L * R * np.pi * 1E6  # area in mm^2 half cylinder area
    mdpt_values_m = 1e-6 * mdpt_values_um

    force = displacement_to_force(mdpt_values_m, E, L, R)
    forceperarea = displacement_to_force_area(mdpt_values_m, E, \
            L, R, area)

    values = {"displacement_px" : mdpt_values_px,
              "displacement_um" : mdpt_values_um,
              "force" : force,
              "force_per_area" : forceperarea}

    d_all = chip_statistics(values, data_disp, dt)

    d_all["units"] = {"displacement_px" : "px",
                     "displacement_um" : "$\mu m$",
                     "force" : "$F$",
                     "force_per_area" : "$F/mm^2$"}


    print("Pillar tracking for " + f_disp + " finished")

    if(save_data):
        save_dictionary(f_disp, "track_pillars", d_all)

    return d_all
