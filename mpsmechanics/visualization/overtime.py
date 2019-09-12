
"""

Åshild Telle / Simula Research Labratory / 2019

"""


import os
import numpy as np
import matplotlib.pyplot as plt

from ..utils.iofuns.data_layer import read_prev_layer
from ..utils.iofuns.folder_structure import make_dir_structure, \
        make_dir_layer_structure
from ..mechanical_analysis.mechanical_analysis import analyze_mechanics
from ..pillar_tracking.pillar_tracking import track_pillars

def plot_over_time(ax, avg_values, std_values, time, intervals, label, unit):
    """

    Args:
        values - 1D numpy array, assumed to be
            relevant values over time, uniformly
            distributed (same time step)
        time - 1D numpy array for time steps
        label - generates file name
        ylabel - y axis label

    """

    if len(intervals) > 0:

        for i in intervals:
            ax.axvline(x=time[i[0]], c='g', alpha=0.8)
        ax.axvline(x=time[i[1]], c='g', alpha=0.8)

    zeros = np.zeros_like(time)
    subvalues = avg_values - std_values
    minvalues = np.where(subvalues < 0, zeros, subvalues)
    maxvalues = avg_values + std_values

    ax.plot(time, avg_values)
    ax.fill_between(time, minvalues, \
            maxvalues, color='gray', alpha=0.5)

    label = (label.replace("_", " ")).capitalize()

    ax.set_ylabel(label + " (" + unit + ")")


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
    
    output_folder = make_dir_layer_structure(f_in, "visualize_over_time")
    
    data = read_prev_layer(f_in, layer_name, layer_fn, \
            save_data=save_data)
    time = data["time"]
    # average over time

    N = len(list(data["over_time_avg"].keys()))

    fig, axs = plt.subplots(N, 1, figsize=(10, 2*N), sharex=True)

    # special one for beatrate and intervals
    
    #for m in data["maxima"]:
    #    axs[0].axvline(x=time[m], c='r')
    #axs[0].set_ylabel("Displacement / Beatrate")
    

    # then every other quantity

    for (ax, key) in zip(axs, data["over_time_avg"].keys()):
        plot_over_time(ax, data["over_time_avg"][key], \
                data["over_time_std"][key], data["time"],
                data["intervals"], key, data["units"][key])

    axs[-1].set_xlabel(r"Time ($ms$)")
    filename = os.path.join(output_folder, "analyze_mechanics.png")
    plt.savefig(filename, dpi=300)
    plt.clf()


def visualize_over_time(f_in, layers, save_data=True):
    """

    Visualize mechanics - "main function"

    """
    layers = layers.split(" ")
    
    for layer in layers:
        layer_fn = eval(layer)
        stats_over_time(f_in, layer, layer_fn, save_data)
    print("Plots finished")
