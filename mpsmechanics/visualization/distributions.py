"""

Åshild Telle / Simula Research Laboratory / 2019

"""

import os
import numpy as np
import matplotlib.pyplot as plt

from mpsmechanics.utils.data_layer import read_prev_layer, generate_filename
from mpsmechanics.dothemaths.operations import calc_norm_over_time
from mpsmechanics.mechanical_analysis.mechanical_analysis import analyze_mechanics
from mpsmechanics.visualization.setup_plots import setup_frame, get_plot_fun, make_pretty_label


def plot_distribution(axis, values, time, value_range, label):
    """

    Plots histogram for 2D data (at a given time step).

    Args:
        axis - axis in subplot instance
        values - X x Y numpy array
        time - which time step (number)
        value_range - min and max meaningful values for this quantity
        label - description

    """

    assert len(values.shape) == 2, \
            "Error: 2D numpy array expected as input."

    axis.set_yscale("log")

    lim = np.max(np.abs(values))
    spacing = 0.1*lim
    axis.set_xlim(max(-lim, value_range[0]) - spacing, \
                  min(lim, value_range[1]) + spacing)
    
    num_bins = 100
    axis.hist(values.flatten(), bins=num_bins, color='#28349C')

    plt.suptitle(f"{label}\nTime: {int(time)} ms")


def plot_1d_values(values, time, time_step, value_range, label):
    """

    Plots histogram for 1D data.

    Args:
        values - T x X x Y numpy array
        time - corresponding time steps; 1D numpy array of length T
        time_step - which time step to plot for
        value_range - min and max meaningful values for this quantity
        label - description

    """

    axes, _ = setup_frame(1, 1, True, True)

    plot_distribution(axes[0], values[time_step], \
                      time[time_step], value_range, label)

    axes[0].set_title(f"Scalar value")
    axes[0].set_xlabel(label)


def plot_2d_values(values, time, time_step, value_range, label):
    """

    Plots histogram for 2D data.

    Args:
        values - T x X x Y x 2 numpy array
        time - corresponding time steps; 1D numpy array of length T
        time_step - which time step to plot for
        value_range - min and max meaningful values for this quantity
        label - description

    """

    axes, _ = setup_frame(1, 2, True, True)

    x_values = values[:, :, :, 0]
    y_values = values[:, :, :, 1]

    for (axis, component) in zip(axes, (x_values, y_values)):
        plot_distribution(axis, component[time_step], \
                          time[time_step], value_range, label)

    axes[0].set_title("x component")
    axes[1].set_title("y component")


def plot_2x2d_values(values, time, time_step, value_range, label):
    """

    Plots histogram for 2D data.

    Args:
        values - T x X x Y x 2 x 2 numpy array
        time - corresponding time steps; 1D numpy array of length T
        time_step - which time step to plot for
        value_range - min and max meaningful values for this quantity
        label - description

    """

    axes, _ = setup_frame(2, 2, True, True)

    ux_values = values[:, :, :, 0, 0]
    uy_values = values[:, :, :, 0, 1]
    vx_values = values[:, :, :, 1, 0]
    vy_values = values[:, :, :, 1, 1]

    all_components = [ux_values, uy_values, vx_values, vy_values]
    subtitles = [r"$u_x$", r"$u_y$", r"$v_x$", r"$v_y$"]

    for (axis, component) in zip(axes, all_components):
        plot_distribution(axis, component[time_step], \
                      time[time_step], value_range, label)

    for (axis, title) in zip(axes, subtitles):
        axis.set_title(title)


def plot_at_peak(values, time, value_range, label, fname):
    """

    Plots histogram at peak value.

    Args:
        values - T x X x Y x D numpy array, D in [(), (2,) or (2, 2)]
        time - corresponding time steps; 1D numpy array of length T
        value_range - min and max meaningful values for this quantity
        label - description
        fname - save plots here

    """

    peak = np.argmax(calc_norm_over_time(values))
    plot_fn = get_plot_fun(values, \
            [plot_1d_values, plot_2d_values, plot_2x2d_values])

    plot_fn(values, time, peak, value_range, label)

    plt.savefig(fname)
    plt.close('all')


def _plot_for_each_key(f_in, mc_data, param_list, overwrite):
    """

    Make histogram plots for distributions over different quantities

    """
    time = mc_data["time"]
    keys = mc_data["all_values"].keys()

    for key in keys:
        fname = generate_filename(f_in, \
                                  f"distribution_{key}", \
                                  param_list[:2],
                                  ".png",
                                  subfolder="distributions")

        if overwrite or not os.path.isfile(fname):
            print("Plots for " + key + " ...")

            label = make_pretty_label(key, mc_data["unit"][key])

            values = mc_data["all_values"][key]
            value_range = mc_data["range"][key]
            plot_at_peak(values, time, value_range, label, fname)


def plot_distributions(f_in, overwrite, overwrite_all, param_list):
    """

    "main function"

    Args:
        f_in - BF / nd2 file
        overwrite - recalculate previous data or not
        param_list - list of lists; 3 sublists. First 2 are passed to
            previous layers if data needs to be recalculated; last gives
            parameters for this script.

    """

    print("Parameters visualize distributions:")
    for key in param_list[2].keys():
        print(" * {}: {}".format(key, param_list[2][key]))

    mc_data = read_prev_layer(
        f_in,
        analyze_mechanics,
        param_list[:-1],
        overwrite_all
    )

    _plot_for_each_key(f_in, mc_data, param_list, overwrite)

    print("Distributions plotted, finishing ..")
