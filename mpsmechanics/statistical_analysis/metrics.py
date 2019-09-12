"""

Åshild Telle / Simula Research Laboratory / 2019

"""

import os
import pandas as pd

from ..utils.iofuns.folder_structure import get_input_properties
from ..utils.iofuns.data_layer import read_prev_layer
from ..mechanical_analysis.mechanical_analysis import analyze_mechanics


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

    fout = os.path.join(os.path.join(path, filename), "metrics.csv")
    pd.DataFrame(metrics_data).to_csv(fout, index=False)

    print(f"Data saved to file {fout}.")
