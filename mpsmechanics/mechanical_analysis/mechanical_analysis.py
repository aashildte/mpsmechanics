# -*- coding: utf-8 -*-
"""

Computes mechanical quantities over space and time.

Åshild Telle / Simula Research Labratory / 2019

"""

import os
from collections import defaultdict
import numpy as np

try:
    import mps
except ImportError:
    print("Can't import mps; can't do mechanical analysis.")
    exit(-1)



from ..motion_tracking.motion_tracking import track_motion
from ..motion_tracking.ref_frame import convert_disp_data, \
        calculate_minmax
from ..motion_tracking.restore_resolution import apply_filter

from ..dothemaths.mechanical_quantities import \
        calc_gradients, calc_principal_strain, calc_gl_strain_tensor, calc_deformation_tensor
from ..dothemaths.angular import calc_projection_fraction
from ..dothemaths.heartbeat import calc_beatrate
from .statistics import chip_statistics

from ..utils.data_layer import read_prev_layer, generate_filename, save_dictionary
from ..visualization.overtime import visualize_over_time

def calc_filter_time(dist):
    """

    Filter dependent on time (different for all time steps).

    """
    return np.any((dist != 0), axis=-1)


def calc_strain_filter(dist, size):
    """

    Filter independent of time (same for all time steps);
    more restrictive than the one used for displacement etc.

    """

    filter_org = np.any((dist != 0), axis=(0, -1))

    filter_new = np.copy(filter_org)

    X, Y = filter_org.shape

    for x in range(X):
        for y in range(Y):
            if not filter_org[x, y]:
                xm2 = x - size if x > size else 0
                xp2 = x + size if (x+size) < X else X-1

                ym2 = y - size if y > size else 0
                yp2 = y + size if (y+size) < Y else Y-1
                filter_new[xm2:xp2+1, ym2:yp2+1] *= False

    return np.broadcast_to(filter_new, dist.shape[:3])


def calc_filter_all(dist):
    """

    Filter independent of time (same for all time steps).

    """

    return np.broadcast_to(np.any((dist != 0),
                                  axis=(0, -1)),
                           dist.shape[:3])


def _calc_mechanical_quantities(displacement, um_per_pixel, block_size, \
        angle, time, strain_filter_size):
    """

    Derived quantities - reshape to match expected data structure
    for derived layers

    Args:
        displacement - displacement data, T x X x Y x 2 numpy array
        scale - scaling factor (pixels to um); dx
        angle - angle chamber is tilted with
        time - all time steps

    Returns:
        dictionary - key: description;
                     value: quantity; unit; filter; value range

    """
    
    displacement = um_per_pixel * displacement

    displacement_minmax = convert_disp_data(
        displacement, calculate_minmax(displacement)
    )

    xmotion = calc_projection_fraction(displacement, angle)
    ymotion = calc_projection_fraction(displacement, np.pi/2 + angle)

    ms_to_s = 1e3
    threshold = 2      # um/s

    velocity = ms_to_s * np.divide(
        np.gradient(displacement, axis=0), \
        np.gradient(time)[:, None, None, None]
    )

    velocity_norm = np.linalg.norm(velocity, axis=-1)
    prevalence = np.where(velocity_norm > threshold*np.ones(velocity_norm.shape),
                    np.ones(velocity_norm.shape), np.zeros(velocity_norm.shape))

    dx = um_per_pixel*block_size

    gradients = calc_gradients(displacement, dx)
    deformation_tensor = calc_deformation_tensor(displacement, dx)
    gl_strain_tensor = calc_gl_strain_tensor(displacement, dx)
    principal_strain = calc_principal_strain(displacement, dx)

    filter_time = calc_filter_time(displacement)
    filter_all = calc_filter_all(displacement)
    filter_strain = calc_strain_filter(displacement, strain_filter_size)

    return {
        "displacement": (
            displacement,
            r"$\mu m$",
            filter_all,
            (0, np.nan),
        ),
        "displacement_maximum_difference": (
            displacement_minmax,
            r"$\mu m$",
            filter_all,
            (0, np.nan),
        ),
        "xmotion": (
            xmotion,
            "-",
            filter_time,
            (0, 1),
        ),
        "ymotion": (
            ymotion,
            "-",
            filter_time,
            (0, 1),
        ),
        "velocity": (
            velocity,
            r"$\mu m / s$",
            filter_all,
            (0, np.nan),
        ),
        "prevalence": (
            prevalence,
            "-",
            filter_all,
            (0, np.nan),
        ),
        "gradients": (
            gradients,
            "-",
            filter_strain,
            (0, np.nan),
        ),
        "deformation_tensor": (
            deformation_tensor,
            "-",
            filter_strain,
            (0, np.nan),
        ),
        "Green-Lagrange_strain_tensor": (
            gl_strain_tensor,
            "-",
            filter_strain,
            (0, np.nan),
        ),
        "principal_strain": (
            principal_strain,
            "-",
            filter_strain,
            (0, np.nan),
        ),
    }


def _calc_beatrate(disp_folded, maxima, intervals, time):

    data = defaultdict(dict)

    beatrate_spatial, beatrate_avg, beatrate_std = \
            calc_beatrate(disp_folded, maxima, intervals, time)

    if len(intervals) == 0:
        data["metrics_max_avg"] = data["metrics_avg_avg"] = \
                data["metrics_max_std"] = data["metrics_avg_std"] = 0
    else:
        data["metrics_max_avg"] = np.max(beatrate_avg)
        data["metrics_avg_avg"] = np.mean(beatrate_avg)
        data["metrics_max_std"] = np.max(beatrate_std)
        data["metrics_avg_std"] = np.mean(beatrate_std)

    return beatrate_spatial, beatrate_avg, beatrate_std, data


def analyze_mechanics(f_in, overwrite, param_list, save_data=True):
    """

    Args:
        f_in - file name; either nd2 or npy file
        save_data - to store values or not; default value True

    Returns:
        dictionary with relevant output values

    """

    filename = generate_filename(f_in, "analyze_mechanics", param_list, ".npy")

    if not overwrite and os.path.isfile(filename):
        print("Previous data exist. Use flag --overwrite / -o to recalculate.")
        return np.load(filename, allow_pickle=True).item()
    
    print("Parameters mechanical analysis:")
    for key in param_list[1].keys():
        print(" * {}: {}".format(key, param_list[1][key]))

    data = read_prev_layer(
        f_in,
        track_motion,
        param_list[:-1],
        overwrite
    )

    mps_data = mps.MPS(f_in)
    angle = data["angle"]
    time = mps_data.time_stamps

    disp_data = apply_filter(data["displacement_vectors"], **param_list[1])
    block_size = data["block_size"]

    print("Calculating mechanical quantities for " + f_in)

    um_per_pixel = mps_data.info["um_per_pixel"]

    values_over_time = \
            _calc_mechanical_quantities(disp_data, um_per_pixel, block_size, \
            angle, time, strain_filter_size=2)

    d_all = chip_statistics(values_over_time)

    d_all["time"] = mps_data.time_stamps

    br_spa, beatrate_avg, beatrate_std, data_beatrate = \
            _calc_beatrate(
                d_all["folded"]["displacement"],
                d_all["maxima"],
                d_all["intervals"],
                d_all["time"],
            )

    d_all["beatrate_spatial"] = br_spa
    d_all["beatrate_avg"] = beatrate_avg
    d_all["beatrate_std"] = beatrate_std
    d_all["range"]["beatrate"] = (0, np.nan)
    d_all["units"]["beatrate"] = "beats/s"

    for k in ["metrics_max_avg",
              "metrics_avg_avg",
              "metrics_max_std",
              "metrics_avg_std"]:
        d_all[k]["beatrate"] = data_beatrate[k]

    print(f"Done calculating mechanical quantities for {f_in}.")

    if save_data:
        save_dictionary(filename, d_all)

    visualize_over_time(d_all, filename[:-4])

    return d_all
