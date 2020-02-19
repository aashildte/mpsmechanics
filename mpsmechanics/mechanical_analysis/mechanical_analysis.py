# -*- coding: utf-8 -*-
"""

Computes mechanical quantities over space and time.

Åshild Telle / Simula Research Labratory / 2019

"""

import os
from collections import defaultdict
import numpy as np
import mps

from mpsmechanics.utils.data_layer import read_prev_layer, \
        generate_filename, save_dictionary
from mpsmechanics.dothemaths.heartbeat import \
        calc_beat_intervals, calc_beat_maxima
from mpsmechanics.dothemaths.operations import calc_norm_over_time
from mpsmechanics.motion_tracking.motion_tracking import \
        track_motion
from mpsmechanics.motion_tracking.restore_resolution import \
        apply_filter

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
                "Error: Inconsistent set of keys in nested" + \
                f"dictionary: {dictionary.keys()}, {def_key_set}."

    dict_swapped = defaultdict(dict)
    for key_l1 in dict_org.keys():
        dictionary = dict_org[key_l1]
        for key_l2 in dictionary.keys():
            dict_swapped[key_l2][key_l1] = dict_org[key_l1][key_l2]

    return dict_swapped


def downsample(org_data, downsampling_factor):

    print("orgiginal shape: ", org_data.shape)
    
    downsampled_data = np.array(org_data[:,::downsampling_factor,::downsampling_factor])

    print("new shape: ", downsampled_data.shape)

    return downsampled_data


def _calc_mechanical_quantities(mps_data, mt_data):
    time = mps_data.time_stamps
    # trunkate:
    time = time[:mt_data["displacement_vectors"].shape[0]]

    um_per_pixel = mps_data.info["um_per_pixel"]

    angle = mt_data["angle"]
    downsampling_factor=8
    dx = um_per_pixel*mt_data["block_size"]/downsampling_factor

    disp_data = downsample(org_data = mt_data["displacement_vectors"], \
                           downsampling_factor = downsampling_factor)
    disp_over_time = np.array(calc_norm_over_time(disp_data))

    maxima = calc_beat_maxima(disp_over_time)
    intervals = calc_beat_intervals(disp_over_time)

    spatial = calc_spatial_metrics(disp_data, time, dx, angle, intervals)
    
    d_all = _swap_dict_keys({**spatial})

    d_all["time"] = mps_data.time_stamps
    d_all["maxima"] = maxima
    d_all["intervals"] = intervals

    return d_all



def analyze_mechanics(f_in, overwrite, overwrite_all, param_list, \
        save_data=True):
    """

    Args:
        f_in - file name; either nd2 or npy file
        overwrite - boolean; overwrite *this* layer
        overwrite_all - boolean; overwrite *all* layers
        param_list - list of parameters changed through the
            command line
        save_data - boolean; save output in npy file or not

    Returns:
        dictionary with relevant output values

    """

    filename = generate_filename(f_in, \
                                 "analyze_mechanics", \
                                 param_list, \
                                 ".npy")
    print("filename: ", filename)

    if not overwrite_all and not overwrite and \
            os.path.isfile(filename):
        print("Previous data exist. Use flag --overwrite / -o " + \
                "to recalculate this layer.")
        print("Use flag --overwrite_all / -oa " + \
                "to recalculate data for all layers.")
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
        params = param_list[1]
    else:
        params = {}

    mechanical_quantities = \
                _calc_mechanical_quantities(mps_data, \
                                            mt_data, \
                                            **params)

    print(f"Done calculating mechanical quantities for {f_in}.")

    if save_data:
        save_dictionary(filename, mechanical_quantities)

    return mechanical_quantities
