
"""

Åshild Telle / Simula Research Labratory / 2019

"""


import numpy as np
import matplotlib.pyplot as plt

from .setup_plots import make_pretty_label

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
    maxima = data["maxima"]

    for maximum in maxima:
        axis.axvline(x=time[maximum], c='r')

    shift = (maxima[1] - maxima[0])//2

    if len(intervals) > 2:
        x_vals = [time[ind] for ind in maxima[:-1] + shift]
        avg = data["over_time_avg"]["beatrate"]
        std = data["over_time_std"]["beatrate"]

        axis.errorbar(x_vals, avg, std, ecolor='gray', fmt=".", capsize=3)
        axis.set_ylabel("Beatrate")


def visualize_over_time(data, filename, extension=".png"):
    """

    Average over time. We can add std too? anything else??

    Args:
        input_file - input file; nd2 or npy file
        filename - save as this

    """

    time = data["time"]
    # average over time

    keys = list(data["all_values"].keys())
    keys.remove("beatrate")             # special case
    num_subplots = len(keys) + 1

    _, axes = plt.subplots(num_subplots, 1, \
            figsize=(14, 3*num_subplots), sharex=True)

    # special one for beatrate

    _plot_beatrate(axes[0], data, time)

    # then every other quantity

    for (axis, key) in zip(axes[1:], keys):
        plot_over_time(axis, data["over_time_avg"][key], \
                data["over_time_std"][key], data["time"], \
                data["range_folded"][key])

        plot_intervals(axis, data["time"], data["intervals"])

        label = make_pretty_label(key, data["unit"][key])
        axis.set_ylabel(label)

    axes[-1].set_xlabel(r"Time ($ms$)")

    filename += extension

    plt.savefig(filename, dpi=500)
    plt.clf()
