"""

Åshild Telle / Simula Research Laboratory / 2019

"""

import os
from collections import defaultdict
import pandas as pd
import numpy as np

from ..utils.iofuns.data_layer import read_prev_layer
from ..mechanical_analysis.mechanical_analysis import \
        analyze_mechanics

def _get_file_info(f_in, doses, pacings, media):

    dose = pacing = medium = None

    for dose in doses:
        if dose in f_in:
            break
        dose = None

    for pacing in pacings:
        if pacing in f_in:
            break
        pacing = None

    for medium in media:
        if medium in f_in:
            break
        medium = None

    error_msg = "Error ({}): ".format(f_in) + \
            "Format structure not as expected." + \
            "Please give a folder where " + \
            "dose is in {}".format(doses) + \
            ", pacing is in {}".format(pacings) + \
            "and medium is in {}".format(media)

    assert (dose in doses and pacing in pacings \
            and medium in media), error_msg

    return dose, pacing, medium


def _read_prev_layer(f_in, debug_mode):
    if not debug_mode:
        try:
            data = read_prev_layer(f_in, \
                                   "analyze_mechanics", \
                                   analyze_mechanics)
        except Exception as exp:
            print(f"Could not run script; error msg: {exp}")
    else:
        data = read_prev_layer(f_in, \
                               "analyze_mechanics", \
                               analyze_mechanics)
    return data


def _extract_metrics(data_metrics, all_data, pacing, dose, chip):
    metrics = list(data_metrics["metrics_max_avg"].keys())
    methods = ["metrics_avg_avg", "metrics_avg_std"]

    for key_metric in metrics:
        for key_method in methods:
            key_f = (pacing, key_metric, key_method[8:])
            if not np.isnan(data_metrics[key_method][key_metric]):
                all_data[key_f][dose][chip] = \
                        data_metrics[key_method][key_metric]

def _read_data(input_files, debug_mode):

    doses = ["Dose0", "Dose1", "Dose2", "Dose3", "Dose4", \
             "Dose5", "Dose6"]
    pacings = ["spont", "1Hz"]
    media = ["SM", "MM"]
    chips = ["MM_1A_", "MM_1B_", "MM_3_", "MM_4A_", "MM_4B_", \
             "MM_4_", "SM_1_", "SM_2A_", "SM_2B_", "SM_3_", "SM_4_"]

    all_data = defaultdict(lambda: \
            defaultdict(lambda: defaultdict(float)))

    for f_in in input_files:
        if "20190903_Dose0" in f_in:
            continue

        dose, pacing, chip = _get_file_info(f_in, doses, \
                                          pacings, chips)
        data_metrics = _read_prev_layer(f_in, debug_mode)

        _extract_metrics(data_metrics, all_data, pacing, dose, chip)

    normalized = _normalize_data(all_data, doses, chips)

    return doses, media, normalized


def _normalize_data(all_data, doses, chips):
    """

    Normalises every statistics to zero dose.

    """

    normalized = defaultdict(lambda: \
            defaultdict(lambda: defaultdict(list)))

    for key in all_data.keys():
        for dose in doses:
            for chip in chips:
                medium = chip.split("_")[0]
                if all_data[key][dose][chip] > 0 and \
                        all_data[key]["Dose0"][chip] > 0:
                    org = all_data[key][dose][chip]
                    norm_factor = all_data[key]["Dose0"][chip]
                    norm_value = org/norm_factor
                    normalized[key][dose][medium].append(norm_value)

    return normalized

def _calc_stats(data_per_dose, doses, medium):
    mean = [np.mean(np.array(data_per_dose[dose][medium])) \
                        for dose in doses]
    std = [np.std(np.array(data_per_dose[dose][medium])) \
                        for dose in doses]
    num_samples = [len(data_per_dose[dose][medium]) \
                        for dose in doses]

    mean = np.array(mean)
    std = np.array(std)
    num_samples = np.array(num_samples)

    sem = std/np.sqrt(num_samples)

    return mean, sem, num_samples


def calculate_stats_chips(input_files, debug_mode, output_folder):
    """

    Args:
        input_files - folder name
        debug_mode - catch expressions or not
        output_folder - save results here

    """

    doses, media, norm_data = _read_data(input_files, debug_mode)

    os.makedirs(output_folder, exist_ok=True)

    for key in norm_data.keys():
        data_per_dose = norm_data[key]
        output_file = os.path.join(output_folder, \
                "_".join(key) + ".csv")

        metrics_data = {}
        metrics_data["Dose"] = [str(x) for x in range(len(doses))]

        for medium in media:
            mean, sem, num_samples = \
                    _calc_stats(data_per_dose, doses, medium)
            metrics_data[medium + "_mean"] = mean
            metrics_data[medium + "_sem"] = sem
            metrics_data[medium + "_n"] = num_samples

        headers = ["Dose"]
        for medium in media:
            headers += ["{}".format(medium)]*3
        print("headers: ", headers)

        pd.DataFrame(metrics_data).to_csv(output_file, \
                index=False, header=headers)
        print(f"Data saved to {output_file}.")
