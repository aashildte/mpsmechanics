
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

    
def get_minmax_values(avg_values, std_values, value_range):
    np.warnings.filterwarnings('ignore')

    def_min = value_range[0]*np.ones(std_values.shape)
    subvalues = avg_values - std_values
    minvalues = np.where(subvalues < value_range[0], def_min, subvalues)

    def_max = value_range[1]*np.ones(std_values.shape)
    subvalues = avg_values + std_values
    maxvalues = np.where(subvalues > value_range[1], def_max, subvalues)

    return minvalues, maxvalues


def plot_over_time(ax, avg_values, std_values, time, intervals, \
        label, unit, value_range):
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

    minvalues, maxvalues = get_minmax_values(avg_values, std_values, value_range)

    ax.plot(time, avg_values)
    ax.fill_between(time, minvalues, \
            maxvalues, color='gray', alpha=0.5)

    label = (label.replace("_", " ")).capitalize()

    ax.set_ylabel(label + " (" + unit + ")")


def stats_over_time(f_in, save_data):
    """

    Average over time. We can add std too? anything else??

    Args:
        f_in - input file; nd2 or npy file
        save_data - boolean value; to be passed to layer_fn (save
            output_values in a 'cache' or not)

    """
    
    output_folder = make_dir_layer_structure(f_in, "visualize_over_time")
    
    data = read_prev_layer(f_in, "analyze_mechanics", analyze_mechanics, save_data)
    
    time = data["time"]
    # average over time

    N = len(list(data["over_time_avg"].keys())) + 1

    fig, axs = plt.subplots(N, 1, figsize=(10, 2*N), sharex=True)

    # special one for beatrate
    
    x_vals = [time[x] for x in [i[0] for i in data["intervals"]] + \
                    [data["intervals"][-1][1]]]
    mean = data["beatrate_avg"]
    std = data["beatrate_std"]

    axs[0].errorbar(x_vals, mean, std, ecolor='gray', fmt=".", capsize=3)
    axs[0].set_ylabel("Beatrate")

    # then every other quantity

    minmax = {"displacement" : (0, np.nan),
              "displacement max diff." : (0, np.nan),
              "velocity" : (0, np.nan),
              "xmotion" : (0, 1),
              "principal strain" : (0, np.nan)}

    for (ax, key) in zip(axs[1:], data["over_time_avg"].keys()):
        plot_over_time(ax, data["over_time_avg"][key], \
                data["over_time_std"][key], data["time"],
                data["intervals"], key, data["units"][key], minmax[key])

    axs[-1].set_xlabel(r"Time ($ms$)")
    filename = os.path.join(output_folder, "analyze_mechanics.png")
    plt.savefig(filename, dpi=500)
    plt.clf()


def visualize_over_time(f_in, save_data=True):
    """

    Visualize mechanics - "main function"

    """
    
    stats_over_time(f_in, save_data)

    print("Plots finished")
