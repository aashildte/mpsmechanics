"""

Åshild Telle / Simula Research Laboratory / 2019

"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict

from ..utils.iofuns.folder_structure import get_input_properties
from ..utils.iofuns.data_layer import read_prev_layer
from ..mechanical_analysis.mechanical_analysis import analyze_mechanics


def calculate_metrics(input_file):
    """

    Calculates / collects metric values from mechanical analysis layer.
    Writes values to a csv file.

    """
    
    path, filename, ext = get_input_properties(input_file)

    assert ext == "nd2", "Error: Wrong file formate"
    assert "BF" in filename, "Error: Not a BF file?"
    
    data = read_prev_layer(input_file, \
                "analyze_mechanics", analyze_mechanics)

    keys = list(data["metrics_max"].keys())
    metrics_data = {}
    metrics_data["Quantities"] = [x.capitalize() for x in keys]
    metrics_data["Maximum value"] = []
    metrics_data["Mean value"] = []

    for key in keys:
        metrics_data["Maximum value"].append(data["metrics_max"][key])
        metrics_data["Mean value"].append(data["metrics_mean"][key])

    fout = os.path.join(os.path.join(path, filename), "metrics.csv")

    df = pd.DataFrame(metrics_data, columns = ["Quantities", "Maximum value", "Mean value"])
    df.to_csv(fout)
