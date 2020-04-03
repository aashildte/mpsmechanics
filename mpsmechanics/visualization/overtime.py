"""

Åshild Telle / Simula Research Labratory / 2019

"""

import numpy as np
import matplotlib.pyplot as plt

from ..utils.bf_mps import BFMPS
from ..utils.data_layer import generate_filename, read_prev_layer
from ..mechanical_analysis.mechanical_analysis import (
    analyze_mechanics,
)
from .setup_plots import make_pretty_label


def get_minmax_values(avg_values, std_values, value_range):
    """

    Calculates avg +- std; cuts of values outside given value range.

    TODO: We can probably split this into two functions.

    """
    np.warnings.filterwarnings("ignore")

    def_min = value_range[0] * np.ones(std_values.shape)
    subvalues = avg_values - std_values
    minvalues = np.where(
        subvalues < value_range[0], def_min, subvalues
    )

    def_max = value_range[1] * np.ones(std_values.shape)
    subvalues = avg_values + std_values
    maxvalues = np.where(
        subvalues > value_range[1], def_max, subvalues
    )

    return minvalues, maxvalues


def plot_intervals(axis, time, intervals):
    """

    Args:
        axis - subplot
        time - 1D numpy array for time steps
        intervals - intervals used to define each beat

    """

    for i in intervals:
        axis.axvline(x=time[i[0]], c="g")
        axis.axvline(x=time[intervals[-1][1]], c="g")


def plot_over_time(
    axis, avg_values, std_values, pacing, time, value_range
):
    """

    Args:
        axis - subplot
        avg_values - 1D numpy array, assumed to be
            relevant values over time, uniformly
            distributed (same time step)
        std_values - same
        pacing - same
        time - 1D numpy array for time steps
        value_range - minimum, maximum range

    """

    minvalues, maxvalues = get_minmax_values(
        avg_values, std_values, value_range
    )

    axis.plot(time, avg_values)
    axis.fill_between(
        time, minvalues, maxvalues, color="gray", alpha=0.5
    )

    if max(pacing > 0.1):  # if chip is actually paced
        pacing = 1 / np.max(pacing) * pacing
        pacing = np.max(maxvalues[~np.isnan(maxvalues)]) * pacing
        axis.plot(time, pacing)


def _make_filename(f_in, param_list):
    fname = generate_filename(
        f_in, f"analyze_mechanics", param_list, ".png", subfolder=""
    )
    return fname


def visualize_over_time(f_in, overwrite, overwrite_all, param_list):
    """

    Args:
        input_file - input file; BF nd2 file
        filename - save as this

    """

    mps_data = BFMPS(f_in)
    pacing = mps_data.pacing

    data = read_prev_layer(f_in, analyze_mechanics)

    metrics = list(data["all_values"].keys())
    time = data["time"]
    intervals = data["intervals"]

    keys = list(data["all_values"].keys())
    num_subplots = len(keys)

    _, axes = plt.subplots(
        num_subplots, 1, figsize=(14, 3 * num_subplots), sharex=True
    )

    for (metric, axis) in zip(metrics, axes):
        avg_values = data["over_time_avg"][metric]
        std_values = data["over_time_std"][metric]
        value_range = data["range_folded"][metric]

        plot_over_time(
            axis, avg_values, std_values, pacing, time, value_range
        )
        plot_intervals(axis, time, intervals)

        label = make_pretty_label(metric, data["unit"][metric])
        axis.set_ylabel(label)

    axes[-1].set_xlabel(r"Time ($ms$)")

    filename = _make_filename(f_in, param_list)

    plt.savefig(filename, dpi=500)
    plt.clf()

    print(f"Plot for metrics over time saved to {filename}.")
