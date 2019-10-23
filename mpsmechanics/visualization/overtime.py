
"""

Åshild Telle / Simula Research Labratory / 2019

"""


import os
import numpy as np
import matplotlib.pyplot as plt

from ..utils.iofuns.data_layer import read_prev_layer
from ..utils.iofuns.folder_structure import get_input_properties
from ..mechanical_analysis.mechanical_analysis import analyze_mechanics


def get_minmax_values(avg_values, std_values, value_range):
    """

    Calculates avg +- std; cuts of values outside given value range.

    TODO: We can probably split this into two functions.

    """
    np.warnings.filterwarnings('ignore')

    def_min = value_range[0]*np.ones(std_values.shape)
    subvalues = avg_values - std_values
    minvalues = np.where(subvalues < value_range[0], def_min, subvalues)

    def_max = value_range[1]*np.ones(std_values.shape)
    subvalues = avg_values + std_values
    maxvalues = np.where(subvalues > value_range[1], def_max, subvalues)

    return minvalues, maxvalues


def plot_intervals(axis, time, intervals):
    """

    Args:
        axis - subplot
        time - 1D numpy array for time steps
        intervals - intervals used to define each beat

    """

    for i in intervals:
        axis.axvline(x=time[i[0]], c='g')
        axis.axvline(x=time[intervals[-1][1]], c='g')


def plot_over_time(axis, avg_values, std_values, time, value_range):
    """

    Args:
        axis - subplot
        avg_values - 1D numpy array, assumed to be
            relevant values over time, uniformly
            distributed (same time step)
        std_values - same
        time - 1D numpy array for time steps
        value_range - minimum, maximum range

    """


    minvalues, maxvalues = get_minmax_values(avg_values, std_values, value_range)

    axis.plot(time, avg_values)
    axis.fill_between(time, minvalues, \
            maxvalues, color='gray', alpha=0.5)


def _plot_beatrate(axis, data, time):
    intervals = data["intervals"]

    if len(intervals) > 3:
        x_vals = [(time[i[0]] + time[i[1]])/2 \
                        for i in intervals[:-1]]
        mean = data["beatrate_avg"]
        std = data["beatrate_std"]

        for i in intervals:
            axis.axvline(x=time[i[0]], c='r')
            axis.axvline(x=time[i[1]], c='r')

        axis.errorbar(x_vals, mean, std, ecolor='gray', fmt=".", capsize=3)
        axis.set_ylabel("Beatrate")


def stats_over_time(f_in, save_data):
    """

    Average over time. We can add std too? anything else??

    Args:
        f_in - input file; nd2 or npy file
        save_data - boolean value; to be passed to layer_fn (save
            output_values in a 'cache' or not)

    """

    path, filename, _ = get_input_properties(f_in)
    output_folder = os.path.join(path, filename, "mpsmechanics")

    data = read_prev_layer(f_in, "analyze_mechanics", \
            analyze_mechanics, save_data)

    time = data["time"]
    # average over time

    num_subplots = len(list(data["over_time_avg"].keys())) + 1

    _, axes = plt.subplots(num_subplots, 1, \
            figsize=(14, 3*num_subplots), sharex=True)

    # special one for beatrate

    _plot_beatrate(axes[0], data, time)

    # then every other quantity

    for (axis, key) in zip(axes[1:], data["over_time_avg"].keys()):
        plot_over_time(axis, data["over_time_avg"][key], \
                data["over_time_std"][key], data["time"], \
                data["range"][key])
        plot_intervals(axis, data["time"], data["intervals"])

        label = (key.replace("_", " ")).capitalize() + " (" + \
                data["units"][key] + ")"
        axis.set_ylabel(label)

    axes[-1].set_xlabel(r"Time ($ms$)")
    filename = os.path.join(output_folder, "analyze_mechanics.png")
    plt.savefig(filename, dpi=500)
    plt.clf()


def visualize_over_time(f_in, save_data=True):
    """

    Visualize mechanics - "main function"

    """

    stats_over_time(f_in, save_data)

    print("Plots finished")
