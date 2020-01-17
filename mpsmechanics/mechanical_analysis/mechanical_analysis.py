# -*- coding: utf-8 -*-
"""

Computes mechanical quantities over space and time.

Åshild Telle / Simula Research Labratory / 2019

"""

import os
from collections import defaultdict
import numpy as np
import mps

from mpsmechanics.utils.data_layer import read_prev_layer, generate_filename, save_dictionary
from mpsmechanics.dothemaths.heartbeat import calc_beat_intervals, calc_beat_maxima
from mpsmechanics.dothemaths.operations import calc_norm_over_time
from mpsmechanics.motion_tracking.motion_tracking import track_motion
from mpsmechanics.motion_tracking.restore_resolution import apply_filter

from .metrics_spatial import calc_spatial_metrics
from .metrics_beatrate import calc_beatrate_metric

def _swap_dict_keys(dict_org):
    """

    Given a nested dictionary, swap layers.

    Args:
        dict_org - dictionary of dictionaries with same
           structure (same keys)

    Returns:
        dictionary of dictionaries with same structure;
            with keys opposite compared to the original

    """

    def_key_set = list(dict_org.values())[0].keys()
    for dictionary in dict_org.values():
        assert dictionary.keys() == def_key_set, \
                f"Error: Inconsistent set of keys in nested dictionary: \
                {dictionary.keys()}, {def_key_set}."

    dict_swapped = defaultdict(dict)
    for key_l1 in dict_org.keys():
        dictionary = dict_org[key_l1]
        for key_l2 in dictionary.keys():
            dict_swapped[key_l2][key_l1] = dict_org[key_l1][key_l2]

    return dict_swapped


def _calc_mechanical_quantities(mps_data, mt_data, type_filter="gaussian", sigma=3):
    time = mps_data.time_stamps
    um_per_pixel = mps_data.info["um_per_pixel"]

    angle = mt_data["angle"]
    dx = um_per_pixel*mt_data["block_size"]

    disp_data = um_per_pixel*apply_filter(mt_data["displacement_vectors"], type_filter, sigma)
    disp_data_folded = calc_norm_over_time(disp_data)
    maxima = calc_beat_maxima(disp_data_folded)
    intervals = calc_beat_intervals(disp_data_folded)

    spatial = calc_spatial_metrics(disp_data, time, dx, angle, intervals)
    beatrate = calc_beatrate_metric(disp_data, time, maxima, intervals)

    d_all = _swap_dict_keys({**spatial, **beatrate})
    d_all["time"] = mps_data.time_stamps
    d_all["maxima"] = maxima
    d_all["intervals"] = intervals

    return d_all



def analyze_mechanics(f_in, overwrite, overwrite_all, param_list, save_data=True):
    """

    Args:
        f_in - file name; either nd2 or npy file
        save_data - to store values or not; default value True

    Returns:
        dictionary with relevant output values

    """

    filename = generate_filename(f_in, "analyze_mechanics", param_list, ".npy")
    print("filename: ", filename)
    if not overwrite_all and not overwrite and os.path.isfile(filename):
        print("Previous data exist. Use flag --overwrite / -o to recalculate this layer.")
        print("Use flag --overwrite_all / -oa to recalculate data for all layers.")
        return np.load(filename, allow_pickle=True).item()

    mps_data = mps.MPS(f_in)
    mt_data = read_prev_layer(
        f_in,
        track_motion,
        param_list[:-1],
        overwrite_all
    )

    print(f"Calculating mechanical quantities for {f_in}")

    if len(param_list) > 1:
        mechanical_quantities = _calc_mechanical_quantities(mps_data, mt_data, **param_list[1])
    else:
        mechanical_quantities = _calc_mechanical_quantities(mps_data, mt_data)

    print(f"Done calculating mechanical quantities for {f_in}.")

    if save_data:
        save_dictionary(filename, mechanical_quantities)

    return mechanical_quantities
