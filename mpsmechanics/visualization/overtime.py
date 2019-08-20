
"""

Åshild Telle / Simula Research Labratory / 2019

"""


import os
import matplotlib.pyplot as plt

from ..utils.iofuns.data_layer import read_prev_layer
from ..utils.iofuns.folder_structure import make_dir_structure, \
        make_dir_layer_structure
from ..mechanical_analysis.mechanical_analysis import analyze_mechanics
from ..pillar_tracking.pillar_tracking import track_pillars

def plot_over_time(values, time, label, unit, path):
    """

    Args:
        values - 1D numpy array, assumed to be
            relevant values over time, uniformly
            distributed (same time step)
        time - 1D numpy array for time steps
        label - generates file name
        ylabel - y axis label

    """

    plt.plot(values)

    label = label.replace("_", " ")

    plt.ylabel(label + " (" + unit + ")")

    filename = os.path.join(path, label) + ".png"
    plt.savefig(filename, dpi=300)
    plt.clf()


def stats_over_time(f_in, layer_name, layer_fn, save_data):
    """

    Average over time. We can add std too? anything else??

    Args:
        f_in - input file; nd2 or npy file
        layer_name - which values to consider
        layer_fn - corresponding function, if not already calculated
        save_data - boolean value; to be passed to layer_fn (save
            output_values in a 'cache' or not)

    """
    
    output_folder = os.path.join(\
            make_dir_layer_structure(f_in, "visualize_over_time"), \
            layer_name)
    make_dir_structure(output_folder)
    
    data = read_prev_layer(f_in, layer_name, layer_fn, \
            save_data=save_data)

    # average over time

    for key in data["over_time_avg"].keys():
        plot_over_time(data["over_time_avg"][key], data["time"],
                       key, data["units"][key], output_folder)


def visualize_over_time(f_in, layers, save_data=True):
    """

    Visualize mechanics - "main function"

    """
    layers = layers.split(" ")
    
    for layer in layers:
        layer_fn = eval(layer)
        stats_over_time(f_in, layer, layer_fn, save_data)
