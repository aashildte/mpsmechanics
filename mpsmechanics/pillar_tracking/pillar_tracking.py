# -*- coding: utf-8 -*-
"""

Point tracking related functions

Åshild Telle / Simula Research Laboratory / 2019

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

def _define_pillars(p_values, tracking_type = 'large_radius', no_meshpts=200):
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
        x_pos, y_pos, radius = p_values[i]

        if tracking_type == "large_radius":
            for j in range(no_meshpts):
                pillars[i, j, 0] = x_pos + radius*np.cos(angles[j])
                pillars[i, j, 1] = y_pos + radius*np.sin(angles[j])

        elif tracking_type == "small_radius":
            small_radius = radius - 9
            for j in range(no_meshpts):
                pillars[i, j, 0] = x_pos + small_radius*np.cos(angles[j])
                pillars[i, j, 1] = y_pos + small_radius*np.sin(angles[j])
        else:
            for j in range(no_meshpts):
                random_radius = np.random.uniform(0, radius)
                pillars[i, j, 0] = x_pos + random_radius*np.cos(angles[j])
                pillars[i, j, 1] = y_pos + random_radius*np.sin(angles[j])

    return pillars


def _calculate_current_timestep(x_coords, y_coords, data_disp, pillars):
    """

    Calculates values at given tracking points (defined by pillars)
    based on interpolation.
    
    # TODO can we make this step faster by only interpolating in
    # the relevant region?

    Args:
        x_coords - x coordinates, dimension X
        y_coords - y coordinates, dimension Y
        data_disp - numpy array of dimensions X x Y x 2
        pillars - numpy array of dimensions no_pillars x no_meshpts x 2

    Returns:
        numpy array of dimension no_meshpts x 2, middle point (average
            of all points on the circumfence); relative displacement

    """

    fn_rel = interpolate_values_2D(x_coords, y_coords, data_disp)

    no_pillars, no_meshpts, no_dims = pillars.shape

    disp_values = np.zeros((no_pillars, no_meshpts, no_dims))
    
    for p in range(no_pillars):
        for n in range(no_meshpts):
            disp_values[p, n] = fn_rel(*pillars[p, n])
    
    return disp_values


def _track_pillars_over_time(data_disp, pillars_mpoints, size_x, size_y,
                             tracking_type, no_meshpts=200):
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

    pillars = _define_pillars(pillars_mpoints, tracking_type=tracking_type, no_meshpts=no_meshpts)

    # some general values
    T, x_dim, y_dim = data_disp.shape[:3]
    no_pillars, no_meshpts = pillars.shape[:2]

    x_coords = np.linspace(0, size_x, x_dim)
    y_coords = np.linspace(0, size_y, y_dim)

    # store data - as an average of all meshpoints
    rel_values = np.zeros((T, no_pillars, no_meshpts, 2))

    # TODO should be done in parallel, using threads
    for t in range(T):
        rel_values[t] = \
                _calculate_current_timestep(x_coords, y_coords, \
                                            data_disp[t], pillars)

    # absolute values: add relative to first positions
    abs_values = rel_values + pillars[None, :, :, :]

    return rel_values, abs_values


def _find_pillar_positions_file(outdir):
    """

    TODO change to npy files

    Args:
        f_disp - displacement data file

    Returns:
        csv file for pillar positions, expected to be in a subfolder
        with the same name

    """

    npy_file = os.path.join(outdir, "pillars.npy")
    csv_file = os.path.join(outdir, "pillars.csv")

    assert os.path.isfile(npy_file) or os.path.isfile(csv_file), \
        "Error: No pillar position file found."

    if os.path.isfile(npy_file):
        return npy_file
    else:
        return csv_file


def track_pillars(f_disp, outdir, L=50E-6, R=10E-6, E=2.63E-6, \
        save_data=True, tracking_type='large_radius', no_meshpts=200, max_motion=3):
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

    f_pts = _find_pillar_positions_file(outdir)

    # displacement data and positions of pillars
    data_disp, scaling_factor, angle, dt, size_x, size_y = \
            read_mt_file(f_disp, outdir, max_motion)
    pillars_mpoints = read_pt_file(f_pts)

    print("Tracking pillars for data set: ", f_disp)
    rel_values_px, abs_values_px = \
            _track_pillars_over_time(data_disp, \
            pillars_mpoints, size_x, size_y, no_meshpts=no_meshpts, tracking_type=tracking_type)
    
    # then do a couple of transformations ..
    rel_values_um = 1/scaling_factor*rel_values_px
    abs_values_um = 1/scaling_factor*abs_values_px

    area = L * R * np.pi * 1E6  # area in mm^2 half cylinder area
    values_m = 1e-6 * rel_values_um

    force = displacement_to_force(values_m, E, L, R)
    forceperarea = displacement_to_force_area(values_m, E, \
            L, R, area)

    values = {"absolute_displacement_px" : abs_values_px,
              "absolute_displacement_um" : abs_values_um,
              "relative_displacement_px" : rel_values_px,
              "relative_displacement_um" : rel_values_um,
              "force" : force,
              "force_per_area" : forceperarea}

    d_all = chip_statistics(values, data_disp, dt)

    d_all["units"] = {"relative_displacement_px" : "px",
                     "relative_displacement_um" : "$\mu m$",
                    "absolute_displacement_px" : "px",
                     "absolute_displacement_um" : "$\mu m$",
                     "force" : "$F$",
                     "force_per_area" : "$F/mm^2$"}

    print("Pillar tracking for " + f_disp + " finished")

    if(save_data):
        save_dictionary(outdir, "track_pillars", d_all)

    return d_all
