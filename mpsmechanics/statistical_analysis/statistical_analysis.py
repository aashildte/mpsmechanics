"""

Åshild Telle / Simula Research Laboratory / 2019

"""

import os
import pandas as pd

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

    assert (d in doses and p in pacing and m in media), error_msg

    return d, p, m


def _read_data(input_files, debug_mode):
    
    doses = ["Dose0", "Dose1", "Dose2", "Dose3", "Dose4", "Dose5", "Dose6"]
    pacings = ["spont", "1Hz"]
    media = ["SM", "MM"]

    all_data = defaultdict(lambda : defaultdict(lambda : defaultdict(lambda : defaultdict(list))))

    for f_in in input_files:

        if "Dose0_1pm" in f_in:        # I'm not sure what this measures
            continue

        dose, pacing, medium = _get_file_info(f_in, doses, pacings, media)

        if not debug_mode:
            try:
                data = read_prev_layer(f_in, "analyze_mechanics", analyze_mechanics)
            except Exception as e:
                print(f"Could not run script; error msg: {e}")
        else:
            data = read_prev_layer(f_in, "analyze_mechanics", analyze_mechanics)

        for key_s1 in list(data["metrics_max_avg"].keys()):
            for key_s2 in ["metrics_max_avg", "metrics_avg_avg", "metrics_max_std", "metrics_avg_std"]:
                key_f = (key_s1, key_s2[8:]) 
                if not np.isnan(data[key_s2][key_s1]):
                    all_data[key_f][dose][pacing][medium].append(data[key_s2][key_s1])

    return doses, pacing, media, all_data


def calculate_stats_chips(input_files, debug_mode, output_folder):

    doses, pacing, media, all_data = _read_data(input_files, debug_mode)

    os.makedirs(output_folder, exist_ok=True)

    for key in all_data.keys():
        data_per_dose = all_data[key] 
        output_file = os.path.join(output_folder, \
                "_".join(key) + ".csv") 

        metrics_data = {}
        metrics_data["Dose"] = [str(x) for x in range(len(doses))]

        for m in media:
            for p in pacing:
                metrics_data[m + "_" + p + "_mean"] = [np.mean(np.array(data_per_dose[d][p][m])) for d in doses]
                metrics_data[m + "_" + p + "_std"] = [np.mean(np.array(data_per_dose[d][p][m])) for d in doses]
                metrics_data[m + "_" + p + "_n"] = [len(data_per_dose[d][p][m]) for d in doses]

        headers = ["Dose"]
        for m in media:
            for p in pacing:
                headers += ["{} / {}".format(m, p)]*3

        pd.DataFrame(metrics_data).to_csv(output_file, index=False, header=headers)
        print(f"Data saved to {output_file}.")
