# -*- coding: utf-8 -*-
"""

Point tracking related functions

Åshild Telle / Simula Research Laboratory / 2019

"""

import os
import numpy as np

import mps

from ..motion_tracking.motion_tracking import track_motion

from ..dothemaths.interpolation import interpolate_values_2D
from ..dothemaths.statistics import chip_statistics

from ..utils.iofuns.folder_structure import get_input_properties 
from ..utils.iofuns.save_values import save_dictionary
from ..utils.iofuns.data_layer import read_prev_layer

from .forcetransformation import displacement_to_force, \
        displacement_to_force_area

def _define_pillars(pillar_positions, tracking_type = 'large_radius', no_tracking_pts=200):
    """

    Defines circle to mark the pillar's circumference, based on
    information about the pillar's initial middle position.

    Args:
        pillar_positions - dictionary with information about pillar positions
        no_tracking_pts - number of mesh points used

    Returns:
        p_values - mesh points on the circumsphere of the circle
            defined by x, y, z
    """

    assert no_tracking_pts > 0, "no_tracking_pts must be greater than 0"
    
    expected_keys = ["positions_transverse", "positions_longitudinal", "radii"]
    error_msg = "Error: Expected pillar positions to be given as a dictionary with " + \
            "{} as keys.".format(expected_keys)

    for e_k in expected_keys:
        assert e_k in pillar_positions.keys(), error_msg

    x_positions = pillar_positions["positions_longitudinal"]
    y_positions = pillar_positions["positions_transverse"]
    radii = pillar_positions["radii"]

    no_pillars = len(x_positions)
    pillars = np.zeros((no_pillars, no_tracking_pts, 2))
    angles = np.linspace(0, 2*np.pi, no_tracking_pts)

    for i in range(no_pillars):
        x_pos, y_pos, radius = x_positions[i], y_positions[i], radii[i]

        if tracking_type == "large_radius":
            for j in range(no_tracking_pts):
                pillars[i, j, 0] = x_pos + radius*np.cos(angles[j])
                pillars[i, j, 1] = y_pos + radius*np.sin(angles[j])

        elif tracking_type == "small_radius":
            small_radius = radius - 9
            for j in range(no_tracking_pts):
                pillars[i, j, 0] = x_pos + small_radius*np.cos(angles[j])
                pillars[i, j, 1] = y_pos + small_radius*np.sin(angles[j])
        else:
            for j in range(no_tracking_pts):
                random_radius = np.random.uniform(0, radius)
                pillars[i, j, 0] = x_pos + random_radius*np.cos(angles[j])
                pillars[i, j, 1] = y_pos + random_radius*np.sin(angles[j])

    return pillars


def _calculate_current_timestep(x_coords, y_coords, disp_data, pillars):
    """

    Calculates values at given tracking points (defined by pillars)
    based on interpolation.
    
    # TODO can we make this step faster by only interpolating in
    # the relevant region?

    Args:
        x_coords - x coordinates, dimension X
        y_coords - y coordinates, dimension Y
        disp_data - numpy array of dimensions X x Y x 2
        pillars - numpy array of dimensions no_pillars x no_tracking_pts x 2

    Returns:
        numpy array of dimension no_tracking_pts x 2, middle point (average
            of all points on the circumfence); relative displacement

    """

    fn_rel = interpolate_values_2D(x_coords, y_coords, disp_data)

    no_pillars, no_tracking_pts, no_dims = pillars.shape

    disp_values = np.zeros((no_pillars, no_tracking_pts, no_dims))
    
    for p in range(no_pillars):
        for n in range(no_tracking_pts):
            disp_values[p, n] = fn_rel(*pillars[p, n])
    
    return disp_values


def _track_pillars_over_time(disp_data, pillar_positions, size_x, size_y,
                             tracking_type, no_tracking_pts=200):
    """

    Tracks position of mesh poinds defined for pillars over time, based
    on the displacement disp_data.

    Args:
        disp_data - displacement data, numpy array of size T x X x Y x 2
        pillar_positions - coordinates + radii of each pillar
        size_x - length (in pixels) of picture
        size_y - width (in pixels) of picture

    Returns:
        numpy array of dimensions T x no_tracking_pts x no_pillars (no_tracking_pts default value);
            gives absolute displacement of circumfence points

    """

    # define pillars by their circumference

    pillars = _define_pillars(pillar_positions, \
                              tracking_type=tracking_type, \
                              no_tracking_pts=no_tracking_pts)

    # some general values
    T, x_dim, y_dim = disp_data.shape[:3]
    no_pillars, no_tracking_pts = pillars.shape[:2]

    x_coords = np.linspace(0, size_x, x_dim)
    y_coords = np.linspace(0, size_y, y_dim)

    # store data - as an average of all meshpoints
    rel_values = np.zeros((T, no_pillars, no_tracking_pts, 2))

    # TODO should be done in parallel, using threads
    for t in range(T):
        rel_values[t] = \
                _calculate_current_timestep(x_coords, y_coords, \
                                            disp_data[t], pillars)

    # absolute values: add relative to first positions
    abs_values = rel_values + pillars[None, :, :, :]

    return rel_values, abs_values


def track_pillars(f_disp, pillar_positions, L=50E-6, R=10E-6, E=2.63E-6, \
        save_data=True, tracking_type='large_radius', no_tracking_pts=200):
    """

    Tracks points corresponding to "pillars" over time.

    Arguments:
        f_disp - filename for displacement data
        pillar_positions - dictionary describing pillar positions + radii 
        L - ??
        R - ??
        E - ??
        save_data - to store values or not; default value True

    Returns:
        dictionary with calculated values

    """

    # displacement data and positions of pillars
    mt_data = read_prev_layer(f_disp, "track_motion", track_motion, \
                    save_data=save_data)

    mps_data = mps.MPS(f_disp)
    disp_data = mt_data["displacement vectors"]
    scaling_factor = mt_data["block size"]*mps_data.info["um_per_pixel"]

    print("Tracking pillars for data set: ", f_disp)
    rel_values_px, abs_values_px = \
            _track_pillars_over_time(disp_data, \
            pillar_positions, mps_data.size_x, mps_data.size_y, \
            no_tracking_pts=no_tracking_pts, tracking_type=tracking_type)
    
    # then do a couple of transformations ..
    rel_values_um = 1/scaling_factor*rel_values_px
    abs_values_um = 1/scaling_factor*abs_values_px

    area = L*R*np.pi*1E6               # area in mm^2 half cylinder area
    values_m = 1e-6*rel_values_um

    force = displacement_to_force(values_m, E, L, R)
    forceperarea = displacement_to_force_area(values_m, E, \
            L, R, area)

    values = {"absolute_displacement_px" : abs_values_px,
              "absolute_displacement_um" : abs_values_um,
              "relative_displacement_px" : rel_values_px,
              "relative_displacement_um" : rel_values_um,
              "force" : force,
              "force_per_area" : forceperarea}

    d_all = {}
    d_all["all_values"] = values
    d_all["units"] = {"relative_displacement_px" : "px",
                     "relative_displacement_um" : "$\mu m$",
                    "absolute_displacement_px" : "px",
                     "absolute_displacement_um" : "$\mu m$",
                     "force" : "$F$",
                     "force_per_area" : "$F/mm^2$"}

    print(f"Pillar tracking for {f_disp} finished")

    if save_data:
        save_dictionary(f_disp, "track_pillars" , d_all)

    return d_all



def track_pillars_init(pillar_design, input_file):
    positions = perform_pillar_detection(pillar_design, input_file, save_values=False)
    track_pillars(input_file, positions)
