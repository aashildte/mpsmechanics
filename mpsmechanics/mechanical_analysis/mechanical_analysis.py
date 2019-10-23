# -*- coding: utf-8 -*-
"""

Computes mechanical quantities over space and time.

Åshild Telle / Simula Research Labratory / 2019

"""

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
from ..dothemaths.mechanical_quantities import \
        calc_principal_strain, calc_gl_strain_tensor
from ..dothemaths.angular import calc_projection_fraction
from ..dothemaths.heartbeat import calc_beatrate
from ..dothemaths.operations import calc_norm_over_time
from ..dothemaths.statistics import chip_statistics
from ..utils.iofuns.save_values import save_dictionary
from ..utils.iofuns.data_layer import read_prev_layer


def calc_filter_time(dist):
    """

    Filter dependent on time (different for all time steps).

    """
    return np.any((dist != 0), axis=-1)


def calc_filter_all(dist):
    """

    Filter independent of time (same for all time steps).

    """
    return np.broadcast_to(np.any((dist != 0),
                                  axis=(0, -1)),
                           dist.shape[:3])


def _calc_mechanical_quantities(displacement, scale, angle, time):
    """

    Derived quantities - reshape to match expected data structure
    for derived layers

    Args:
        displacement - displacement data, T x X x Y x 2 numpy array
        scale - scaling factor (pixels to um)
        angle - angle chamber is tilted with
        time - all time steps

    Returns:
        dictionary - key: description;
                     value: quantity; unit; filter; value range

    """

    displacement = scale * displacement

    displacement_minmax = convert_disp_data(
        displacement, calculate_minmax(displacement)
    )

    xmotion = calc_projection_fraction(displacement, angle)
    ymotion = calc_projection_fraction(displacement, np.pi/2 + angle)

    print("angles: ", angle, np.pi/2 + angle)

    ms_to_s = 1e3
    threshold = 2      # um/s

    velocity = ms_to_s * np.divide(
        np.gradient(displacement, axis=0), \
        np.gradient(time)[:, None, None, None]
    )

    velocity_norm = np.linalg.norm(velocity, axis=-1)
    prevalence = np.where(velocity_norm > threshold*np.ones(velocity_norm.shape),
                    np.ones(velocity_norm.shape), np.zeros(velocity_norm.shape))
    prevalence = prevalence[:,:,:,None]

    gl_strain_tensor = calc_gl_strain_tensor(displacement, scale)
    principal_strain = calc_principal_strain(displacement, scale)

    filter_time = calc_filter_time(displacement)
    filter_all = calc_filter_all(displacement)

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
        "gl_strain_tensor": (
            gl_strain_tensor,
            "-",
            filter_all,
            (0, np.nan),
        ),
        "principal_strain": (
            principal_strain,
            "-",
            filter_all,
            (0, np.nan),
        ),
    }


def _calc_beatrate(disp_folded, maxima, intervals, time):

    data = defaultdict(dict)

    beatrate_spatial, beatrate_avg, beatrate_std = \
            calc_beatrate(disp_folded, maxima, intervals, time)

    if len(beatrate_avg)==0:
        data["metrics_max_avg"] = data["metrics_avg_avg"] = \
                data["metrics_max_std"] = data["metrics_avg_std"] = 0
    else:
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

    data = read_prev_layer(
        input_file,
        "track_motion",
        track_motion,
        save_data=save_data
    )

    mt_data = mps.MPS(input_file)
    disp_data = data["displacement vectors"]
    angle = data["angle"]
    time = mt_data.time_stamps
    scale = data["block size"] * mt_data.info["um_per_pixel"]

    print("Calculating mechanical quantities for " + input_file)

    values_over_time = \
            _calc_mechanical_quantities(disp_data, scale,
                                        angle, time)
    d_all = chip_statistics(values_over_time)

    # TODO include filter in metadata

    d_all["time"] = mt_data.time_stamps

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

    print("Done calculating mechanical quantities for " + \
            input_file)

    if save_data:
        save_dictionary(input_file, "analyze_mechanics", d_all)

    return d_all
