# -*- coding: utf-8 -*-
"""

Point tracking related functions

Åshild Telle / Simula Research Laboratory / 2019

"""

import os
import numpy as np
import mps
import pandas

# from pillar_detection.utility import perform_pillar_detection

from ..mechanical_analysis.mechanical_analysis import analyze_mechanics
from ..dothemaths.interpolation import interpolate_values_xy

from ..utils.folder_structure import get_input_properties
from ..utils.data_layer import read_prev_layer, save_dictionary, generate_filename

from .forcetransformation import displacement_to_force, displacement_to_force_area


def _define_pillars(pillar_positions, radius, no_tracking_pts=200):
    """

    Defines circle to mark the pillar's circumference, based on
    information about the pillar's initial middle position.

    For each pillar, we generate random points on a square around the pillar.
    If the points are not within the given radius we do it again. That way we
    get no_tracking_pts points within the radius, uniformly distributed.

    Args:
        pillar_positions - dictionary with information about pillar positions
        radius
        no_tracking_pts - number of mesh points used

    Returns:
        p_values - mesh points on the circumsphere of the circle
            defined by x, y, z
    """

    assert no_tracking_pts > 0, "no_tracking_pts must be greater than 0"

    expected_keys = ["positions_transverse", "positions_longitudinal"]
    error_msg = (
        "Error: Expected pillar positions to be given as a dictionary with "
        + f"{expected_keys} as keys."
    )

    for e_k in expected_keys:
        assert e_k in pillar_positions.keys(), error_msg

    x_positions = pillar_positions["positions_longitudinal"]
    y_positions = pillar_positions["positions_transverse"]

    no_pillars = len(x_positions)
    pillars = np.zeros((no_pillars, no_tracking_pts, 2))

    for i in range(no_pillars):
        x_pos, y_pos = x_positions[i], y_positions[i]

        for j in range(no_tracking_pts):
            random_xpos = np.random.uniform(x_pos, x_pos + radius)
            random_ypos = np.random.uniform(y_pos, y_pos + radius)

            while (random_xpos - x_pos) ** 2 + (random_ypos - y_pos) ** 2 > radius ** 2:
                random_xpos = np.random.uniform(x_pos, x_pos + radius)
                random_ypos = np.random.uniform(y_pos, y_pos + radius)

            pillars[i, j, 0] = random_xpos
            pillars[i, j, 1] = random_ypos

    return pillars


def _calculate_current_timestep(x_coords, y_coords, disp_data, pillars):
    """

    Calculates values at given tracking points (defined by pillars)
    based on interpolation.

    Args:
        x_coords - x coordinates, dimension X
        y_coords - y coordinates, dimension Y
        disp_data - numpy array of dimensions X x Y x 2
        pillars - numpy array of dimensions no_pillars x no_tracking_pts x 2

    Returns:
        numpy array of dimension no_tracking_pts x 2, middle point (average
            of all points on the circumfence); relative displacement

    """

    fn_rel = interpolate_values_xy(x_coords, y_coords, disp_data)

    no_pillars, no_tracking_pts, no_dims = pillars.shape

    disp_values = np.zeros((no_pillars, no_tracking_pts, no_dims))

    for p in range(no_pillars):
        for n in range(no_tracking_pts):
            disp_values[p, n] = fn_rel(*pillars[p, n])

    return disp_values


def _track_pillars_over_time(
    disp_data, pillar_positions, radius, size_x, size_y, no_tracking_pts=200
):
    """

    Tracks position of mesh poinds defined for pillars over time, based
    on the displacement disp_data.

    Args:
        disp_data - displacement data, numpy array of size T x X x Y x 2
        pillar_positions - coordinates + radii of each pillar
        radius
        size_x - length (in pixels) of picture
        size_y - width (in pixels) of picture
        no tracking points

    Returns:
        numpy array of dimensions T x no_tracking_pts x no_pillars
            (no_tracking_pts default value);
            gives absolute displacement of circumfence points

    """

    # define pillars by their circumference

    pillars = _define_pillars(pillar_positions, radius, no_tracking_pts)

    # some general values
    T, x_dim, y_dim = disp_data.shape[:3]
    no_pillars, no_tracking_pts = pillars.shape[:2]

    x_coords = np.linspace(0, size_x, x_dim)
    y_coords = np.linspace(0, size_y, y_dim)

    # store data - as an average of all meshpoints
    rel_values = np.zeros((T, no_pillars, no_tracking_pts, 2))

    for t in range(T):
        rel_values[t] = _calculate_current_timestep(
            x_coords, y_coords, disp_data[t], pillars
        )

    return np.mean(rel_values, axis=2)


def disp_to_force_data(displacement, L=50e-6, R=10e-60, E=2.63e6):
    area = L * R * np.pi * 1e6  # area in mm^2 half cylinder area
    values_m = 1e-6 * displacement  # um -> m

    force = displacement_to_force(values_m, E, L, R)
    force_per_area = displacement_to_force_area(values_m, E, L, R, area)

    return force, force_per_area


def read_pillar_positions(f_in):

    path, filename, _ = get_input_properties(f_in)
    pos_file = os.path.join(path, filename, "mpsmechanics", "pillar_tracking", "pillar_positions.csv")

    assert os.path.isfile(
        pos_file
    ), f"Error: Expected pillar position file located in {pos_file}."

    positions = pandas.read_csv(pos_file).to_dict("list")

    positions_flipped = {}
    positions_flipped["positions_longitudinal"] = positions["Y"]
    positions_flipped["positions_transverse"] = positions["X"]

    return positions_flipped


def track_pillars(f_in, overwrite, overwrite_all, param_list, save_data=True):
    """

    Tracks points corresponding to "pillars" over time.

    Arguments:
        f_in - filename for displacement data
        pillar_positions - dictionary describing pillar positions + radii 

    Returns:
        dictionary with calculated values

    """

    # displacement data and positions of pillars

    mps_data = mps.MPS(f_in)
    um_per_pixel = mps_data.info["um_per_pixel"]

    mt_data = read_prev_layer(f_in, analyze_mechanics)
    disp_data = mt_data["all_values"]["displacement"] / um_per_pixel  # in pixels

    pillar_positions = read_pillar_positions(f_in)  # in pixels

    print(f"Tracking pillars for {f_in}")

    radius = 10

    disp_over_time = um_per_pixel * _track_pillars_over_time(
        disp_data, pillar_positions, radius, mps_data.size_x, mps_data.size_y
    )

    force, force_per_area = disp_to_force_data(disp_over_time, R=radius * 1e-6)

    values = {
        **pillar_positions,
        "displacement_um": disp_over_time,
        "force": force,
        "force_per_area": force_per_area,
    }

    print(f"Pillar tracking for {f_in} finished")

    if save_data:
        filename = generate_filename(f_in, "track_pillars", param_list, ".npy", subfolder="pillar_tracking")
        save_dictionary(filename, values)

    return values
