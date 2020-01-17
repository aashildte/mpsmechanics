"""

Åshild Telle / Simula Research Laboratory / 2019

"""

import os
import pandas as pd

from ..utils.folder_structure import get_input_properties
from ..utils.data_layer import read_prev_layer
from ..mechanical_analysis.mechanical_analysis import analyze_mechanics


def _data_to_dict(f_in, param_list, overwrite_all):
    data = read_prev_layer(
        f_in,
        analyze_mechanics,
        param_list[:-1],
        overwrite_all
    )

    keys = list(data["metrics_max_avg"].keys())
    
    metrics_data = {}
    metrics_data[" "] = []
    metrics_data["Maximum average value"] = []
    metrics_data["Average average value"] = []
    metrics_data["Maximum standard deviation"] = []
    metrics_data["Average standard deviation"] = []

    for key in keys:
        key_m = key.replace("_", " ").capitalize()
        metrics_data[" "].append(key_m)
        metrics_data["Maximum average value"].append(data["metrics_max_avg"][key])
        metrics_data["Average average value"].append(data["metrics_avg_avg"][key])
        metrics_data["Maximum standard deviation"].append(data["metrics_max_std"][key])
        metrics_data["Average standard deviation"].append(data["metrics_avg_std"][key])
    
    return metrics_data


def _calculate_metrics_file(f_in, metrics_all, data_keys, \
        param_list, overwrite_all):
    path, filename, ext = get_input_properties(f_in)

    metrics_data = _data_to_dict(f_in, param_list, overwrite_all)
    metrics_all["Filename"] += [f_in] + \
            [" "]*(len(metrics_data[" "]))

    assert list(metrics_data.keys()) == data_keys

    for k in data_keys:
       metrics_all[k] += metrics_data[k] + [" "]


def calculate_metrics(f_in, overwrite, overwrite_all, param_list, save_data=True):
    """

    Calculates / collects metric values from mechanical analysis layer.
    Writes values to a csv file.

    Args:
        f_in - nd2 brightfield file

    """

    path, filename, ext = get_input_properties(f_in)

    metrics_data = _data_to_dict(f_in, param_list, overwrite_all)

    folder = os.path.join(path, filename, "mpsmechanics")
    fout = os.path.join(folder, "metrics.csv")
    pd.DataFrame(metrics_data).to_csv(fout, index=False)

    print(f"Data saved to file {fout}.")
