"""

Åshild Telle / Simula Research Laboratory / 2019

"""

import os
import pandas

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

from ..utils.iofuns.folder_structure import get_input_properties
from ..utils.iofuns.data_layer import read_prev_layer
from ..pillar_tracking.pillar_tracking import track_pillars
from ..mechanical_analysis.mechanical_analysis import analyze_mechanics


def _get_file_info(f_in, doses, pacing, media):


    for d in doses:
        if d in f_in:
            break
        d = None

    for p in pacing:
        if p in f_in:
            break
        p = None

    for m in media:
        if m in f_in:
            break
        n = None

    error_msg = "Error ({}): Folder structure not as expected. ".format(f_in) + \
            "Please give a folder s.t. where dose is in {}".format(doses) + \
            ", pacing is in {} and medium is in {}".format(pacing, media)

    assert (d is not None and p is not None and m is not None), error_msg

    return d, (p, m)


def _read_data(input_files, debug_mode):
    
    doses = ["Dose0", "Dose1", "Dose2", "Dose3", "Dose4", "Dose5", "Dose6"]
    pacing = ["spont", "1Hz"]
    media = ["SM", "MM"]

    all_data = defaultdict(dict)

    for f_in in input_files:

        if "Dose0_1pm" in f_in:        # I'm not sure what this measures
            continue

        dose, key_pref = _get_file_info(f_in, doses, pacing, media)
 
        if not debug_mode:
            try:
                data = read_prev_layer(f_in, "analyze_mechanics", analyze_mechanics)
            except Exception as e:
                print(f"Could not run script; error msg: {e}")
        else:
            data = read_prev_layer(f_in, "analyze_mechanics", analyze_mechanics)

        for key_s1 in list(data["metrics_max_avg"].keys()):
            for key_s2 in ["metrics_max_avg", "metrics_avg_avg", "metrics_max_std", "metrics_avg_std"]:
                key = key_pref + (key_s1, key_s2[8:]) 
                all_data[key][dose] = data[key_s2][key_s1]

    return doses, pacing, media, all_data


def calculate_stats_chips(input_files, debug_mode, output_folder):

    doses, pacing, media, all_data = _read_data(input_files, debug_mode)

    for key in all_data.keys():
        data_per_dose = defaultdict(list)

        for d in doses:
            if not np.isnan(all_data[key][d]):
                data_per_dose[d].append(all_data[key][d])

        output_file = os.path.join(output_folder, key.join("_") + ".csv")

        metrics_data = {}
        metrics_data

        pd.DataFrame(metrics_data).to_csv(output_file, index=False)


        print(key)
        print(avg_per_dose)
