"""

Åshild Telle / Simula Research Laboratory / 2019

"""

import os
import pandas as pd

from ..utils.iofuns.folder_structure import get_input_properties
from ..utils.iofuns.data_layer import read_prev_layer
from ..mechanical_analysis.mechanical_analysis import analyze_mechanics


def _data_to_dict(input_file):
    data = read_prev_layer(input_file, \
                "analyze_mechanics", analyze_mechanics)

    keys = list(data["metrics_max_avg"].keys())
    
    metrics_data = {}
    metrics_data[" "] = []
    metrics_data["Maximum average value"] = []
    metrics_data["Average average value"] = []
    metrics_data["Maximum standard deviation"] = []
    metrics_data["Average standard deviation"] = []

    for key in keys:
        metrics_data[" "].append(key.capitalize())
        metrics_data["Maximum average value"].append(data["metrics_max_avg"][key])
        metrics_data["Average average value"].append(data["metrics_avg_avg"][key])
        metrics_data["Maximum standard deviation"].append(data["metrics_max_std"][key])
        metrics_data["Average standard deviation"].append(data["metrics_avg_std"][key])

    return metrics_data


def _calculate_metrics_file(input_file, metrics_all, data_keys):
    path, filename, ext = get_input_properties(input_file)

    metrics_data = _data_to_dict(input_file)
    metrics_all["Filename"] += [input_file] + \
            [" "]*(len(metrics_data[" "]))

    assert list(metrics_data.keys()) == data_keys

    for k in data_keys:
       metrics_all[k] += metrics_data[k] + [" "]


def calculate_metrics_all(input_files, debug_mode):

    data_keys = [" ", "Maximum average value", "Average average value", \
            "Maximum standard deviation", "Average standard deviation"]

    metrics_all = {}

    metrics_all["Filename"] = []

    for k in data_keys:
        metrics_all[k] = []

    for f in input_files:
        if debug_mode:
            _calculate_metrics_file(f, metrics_all, data_keys)
       
        else:
            try:
                _calculate_metrics_file(f, metrics_all, data_keys)
            except Exception as e:
                print(f"Could not find metrics for {f}; error msg: {e}")

    fout = "metrics_summary.csv"
    pd.DataFrame(metrics_all).to_csv(fout, index=False)

    print(f"Data saved to file {fout}.")


def calculate_metrics(input_file):
    """

    Calculates / collects metric values from mechanical analysis layer.
    Writes values to a csv file.

    Args:
        input_file - nd2 brightfield file

    """

    path, filename, ext = get_input_properties(input_file)

    assert ext == "nd2", "Error: Wrong file formate"
    assert "BF" in filename, "Error: Not a BF file?"

    metrics_data = _data_to_dict(input_file)

    fout = os.path.join(os.path.join(path, filename), "metrics.csv")
    pd.DataFrame(metrics_data).to_csv(fout, index=False)

    print(f"Data saved to file {fout}.")
