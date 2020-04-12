# -*- coding: utf-8 -*-
"""

Point tracking related functions

Åshild Telle / Simula Research Laboratory / 2019

"""

import os
import numpy as np
import pandas


from ..utils.bf_mps import BFMPS

from ..mechanical_analysis.mechanical_analysis import analyze_mechanics
from ..dothemaths.interpolation import interpolate_values_xy
from ..dothemaths.operations import calc_norm_over_time

from ..utils.folder_structure import get_input_properties
from ..utils.data_layer import (
    read_prev_layer,
    save_dictionary,
    generate_filename,
)

from .forcetransformation import (
    displacement_to_force,
    displacement_to_force_area,
)

def _calc_max_over_avg_interval(intervals, original_trace):

    trace_per_interval = [
        original_trace[i1:i2] for (i1, i2) in intervals
    ]
    shortest_interval = min(
        [len(trace) for trace in trace_per_interval]
    )
    equal_intervals = [
        trace[:shortest_interval] for trace in trace_per_interval
    ]

    avg_trace = np.mean(np.array(equal_intervals), axis=0)

    return np.max(avg_trace, axis=0)


def calc_int_avg(values, intervals):

    norm = np.linalg.norm(values, axis=2) #will give the norm of the force sqrt(x^2+y^2) -->

    if len(intervals) > 1:
        metrics_int_avg = _calc_max_over_avg_interval(intervals, norm)
    else:
        metrics_int_avg = np.max(norm)

    return metrics_int_avg

def _define_pillars(
    pillar_positions: np.ndarray, radius: float, no_tracking_pts: float = 200,
) -> np.ndarray:
    """

    Defines circle to mark the pillar's circumference, based on
    information about the pillar's initial middle position.

    For each pillar, we generate random points on a square around the pillar.
    If the points are not within the given radius we do it again. That way we
    get no_tracking_pts points within the radius, uniformly distributed.

    Args:
        pillar_positions - numpy array with information about pillar positions
        radius - of each pillar
        no_tracking_pts - number of mesh points used

    Returns:
        p_values - mesh points on the circumsphere of the circle;
            numpy array of dimension
                number of pillars x number of tracking points x 2
    """

    assert no_tracking_pts > 0, "no_tracking_pts must be greater than 0"

    x_positions = pillar_positions[:, 0]
    y_positions = pillar_positions[:, 1]

    no_pillars = len(x_positions)
    pillars = np.zeros((no_pillars, no_tracking_pts, 2))

    for i in range(no_pillars):

        if not np.isnan(x_positions[i]):
            x_pos, y_pos = x_positions[i], y_positions[i]

            for j in range(no_tracking_pts):
                random_xpos = np.random.uniform(x_pos, x_pos + radius)
                random_ypos = np.random.uniform(y_pos, y_pos + radius)

                while (random_xpos - x_pos) ** 2 + (random_ypos - y_pos) ** 2 > radius ** 2:
                    random_xpos = np.random.uniform(x_pos, x_pos + radius)
                    random_ypos = np.random.uniform(y_pos, y_pos + radius)

                pillars[i, j, 0] = random_xpos
                pillars[i, j, 1] = random_ypos

        if np.isnan(x_positions[i]):
            for j in range(no_tracking_pts):
                pillars[i,j,0] = np.nan
                pillars[i,j,1] = np.nan

    return pillars


def _calculate_current_timestep(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    disp_data: np.ndarray,
    pillars: np.ndarray,
) -> np.ndarray:
    """

    Calculates values at given tracking points (defined by pillars)
    based on interpolation.

    Args:
        x_coords - x coordinates, dimension X
        y_coords - y coordinates, dimension Y
        disp_data - numpy array of dimensions X x Y x 2
        pillars - numpy array of dimensions no_pillars x no_tracking_pts x 2

    Returns:
        numpy array of dimension no_tracking_pts x 2, middle point; relative displacement

    """
    fn_rel = interpolate_values_xy(x_coords, y_coords, disp_data)

    no_pillars, no_tracking_pts, no_dims = pillars.shape

    disp_values = np.zeros((no_pillars, no_tracking_pts, no_dims))

    for _p in range(no_pillars):
        for _n in range(no_tracking_pts):
            if np.isnan(pillars[_p, :, :]).any():
                #print(pillars[_p, :, :])
                disp_values[_p, _n] = [np.nan, np.nan]
            else:
                disp_values[_p, _n] = fn_rel(*pillars[_p, _n])

    return disp_values


def _track_pillars_over_time(
    disp_data: np.ndarray,
    reference_index: int,
    pillar_positions: np.ndarray,
    radius: float,
    size_x: int,
    size_y: int,
    no_tracking_pts=200,
) -> np.ndarray:
    """

    Tracks position of mesh poinds defined for pillars over time, based
    on the displacement disp_data.

    Args:
        disp_data - displacement data, numpy array of size T x X x Y x 2
        reference_index - index used as reference for displacement data
        pillar_positions - coordinates + radii of each pillar
        radius - of each pillar
        size_x - length (in pixels) of picture
        size_y - width (in pixels) of picture
        no tracking points - number of tracking points per pillar

    Returns:
        numpy array of dimensions T x no_pillars x 2; relative displacement
            of circumfence points

    """

    disp_relative_to_zero = disp_data - disp_data[0]

    # define pillars by their circumference
    #print(pillar_positions)
    pillars = _define_pillars(pillar_positions, radius, no_tracking_pts)
    # some general values
    t_dim, x_dim, y_dim = disp_data.shape[:3]
    no_pillars, no_tracking_pts = pillars.shape[:2]

    x_coords = np.linspace(0, size_x, x_dim)
    y_coords = np.linspace(0, size_y, y_dim)

    # store data - as an average of all meshpoints
    rel_values = np.zeros((t_dim, no_pillars, no_tracking_pts, 2))

    for _t in range(t_dim):
        rel_values[_t] = _calculate_current_timestep(
            x_coords, y_coords, disp_relative_to_zero[_t], pillars
        )

    rel_values = rel_values - rel_values[reference_index]  # relative to reference index

    return np.mean(rel_values, axis=2)


def read_pillar_positions(f_in: str):
    """

    Reads in pillar positions from csv file. Positions
    expected to be given in pixels.

    Args:
        f_in - path to BF nd2 file

    Returns:
        numpy array of dimension number of pillars x 2,
            where the first axis gives x position (along chamber)
            and the second axis gives y position (across chamber)

    """

    path, filename, _ = get_input_properties(f_in)
    pos_file = os.path.join(path, filename, "mpsmechanics", "pillar_positions.csv",)

    assert os.path.isfile(
        pos_file
    ), f"Error: Expected pillar position file located in {pos_file}."

    positions_dict = pandas.read_csv(pos_file).to_dict("list")

    num_pillars = len(positions_dict["X"])

    positions = np.zeros((num_pillars, 2))

    positions[:, 0] = positions_dict["Y"]  # flipped: trans.
    positions[:, 1] = positions_dict["X"]  # flipped: long.

    return positions


def get_displacement_data(f_in, param_list, motion_scaling_factor=1):
    mt_data = read_prev_layer(f_in, analyze_mechanics, param_list[:-1])

    reference_index = mt_data["reference_index"]
    um_per_pixel = mt_data["info"]["um_per_pixel"]
    size_x = mt_data["info"]["size_x"]
    size_y = mt_data["info"]["size_y"]

    disp_data = (1 / um_per_pixel) * mt_data["all_values"]["displacement"]

    disp_data *= motion_scaling_factor

    return reference_index, disp_data, um_per_pixel, size_x, size_y


def track_pillars(
    f_in: str,
    overwrite: bool,
    overwrite_all: bool,
    param_list: dict,
    save_data: bool = True,
) -> dict:
    """

    Tracks points corresponding to "pillars" over time.

    Arguments:
        f_in - filename for displacement data
        pillar_positions - dictionary describing pillar positions + radii

    Returns:
        dictionary with calculated values

    """

    # displacement data and positions of pillars

    pillar_positions = read_pillar_positions(f_in)  # in pixels

    reference_index, disp_data, um_per_pixel, size_x, size_y = get_displacement_data(
        f_in, param_list[:2], **param_list[2]
    )

    print(f"Tracking pillars for {f_in}")

    radius = 10  # um
    height = 55  # um
    elastic_modulus = 2.63e6  # Pa

    over_time_pixels = _track_pillars_over_time(
        disp_data,
        reference_index,
        pillar_positions,
        radius / um_per_pixel,
        size_x,
        size_y,
    )

    over_time_um = um_per_pixel * over_time_pixels

    force = displacement_to_force(
        displacement=over_time_um,
        height=height,
        elastic_modulus=elastic_modulus,
        radius=radius,
    )
    force_per_area = displacement_to_force_area(
        force=force, height=height, radius=radius
    )

    values = {
        "initial_positions": pillar_positions,
        "displacement_pixels": over_time_pixels,
        "displacement_um": over_time_um,
        "force": force,
        "force_per_area": force_per_area,
        "material_parameters": {
            "radius": radius,
            "elastic_modulus": elastic_modulus,
            "height": height,
        },
        "reference_index": reference_index,
    }

    print(f"Pillar tracking for {f_in} finished")

    if save_data:
        filename = generate_filename(f_in, "track_pillars", param_list, ".npy")
        save_dictionary(filename, values)

    return values
