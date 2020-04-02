"""

Åshild Telle / Simula Research Laboratory / 2019

"""


import numpy as np
import matplotlib.pyplot as plt
import mps
import mpsmechanics as mm
import glob
import os
import sys
from argparse import ArgumentParser

from ..dothemaths.operations import calc_norm_over_time
from ..utils.data_layer import generate_filename

from .animation_funs import (
    make_animation,
    get_animation_configuration,
)

from .setup_plots import generate_filenames_pngmp4, make_heatmap_plot, setup_frame


def _set_ax_units(axis, scale, shift):
    axis.set_aspect("equal")
    num_yticks = 8
    num_xticks = 4

    y_range = axis.get_ylim()
    x_range = axis.get_xlim()

    y_to = y_range[0]
    x_to = x_range[1]

    yticks = np.linspace(0, y_to, num_yticks)
    ylabels = scale * yticks + shift

    xticks = np.linspace(0, x_to, num_xticks)
    xlabels = scale * xticks + shift

    xlabels = xlabels.astype(int)
    ylabels = ylabels.astype(int)

    axis.set_yticks(yticks)
    axis.set_xticks(xticks)
    axis.set_yticklabels(ylabels)
    axis.set_xticklabels(xlabels)

    axis.set_xlabel(r"$\mu m$")
    axis.set_ylabel(r"$\mu m$")


def _init_subplots(images, values, time_step, cb_range):
    axes, fig = setup_frame(1, 2, False, False)
    subplots = []

    vmin = np.min(values)
    vmax = np.max(values)

    # overwrite if given by user:

    if cb_range[0] is not None:
        vmin = cb_range[0]

    if cb_range[1] is not None:
        vmax = cb_range[1]

    colormap = "viridis"

    subplots.append(axes[0].imshow(images[time_step], cmap="gray"))
    subplots.append(make_heatmap_plot(axes[1], values[time_step], vmin, vmax, colormap))

    return axes, fig, subplots


def _make_plot_pretty(fig, axes, subplots, subtitles, um_per_pixel, block_size):
    fig.colorbar(subplots[1], ax=axes[1]).set_label("Light intensity")

    for (axis, subtitle) in zip(axes, subtitles):
        axis.set_title(subtitle)

    _set_ax_units(axes[0], um_per_pixel, 0)
    _set_ax_units(axes[1], block_size * um_per_pixel, block_size // 2)


def plot_org_vs_corrected(values, images, um_per_pixel, time, time_step, cb_range):
    """

    Plots original image and magnitude.

    Args:
        images : original images : T x X x Y numpy array
        values : T x X x Y numpy array
        time - corresponding time units
        time_step - integer value; time step of interest, typically peak or 0

    """
    subtitles = [
        "Original image",
        "Pixel intensity, relative to baseline",
    ]

    block_size = images.shape[1] // values.shape[1]

    axes, fig, subplots = _init_subplots(images, values, time_step, cb_range)
    _make_plot_pretty(fig, axes, subplots, subtitles, um_per_pixel, block_size)

    plt.suptitle("Time: {} ms".format(int(time[time_step])))

    def _update(index):
        subplots[0].set_array(images[index])
        subplots[1].set_data(values[index])
        plt.suptitle("Time: {} ms".format(int(time[index])))

    return fig, _update


def _plot_at_time_step(values, images, um_per_pixel, time, time_step, cb_range, fname):

    if time_step is None:
        time_step = np.argmax(calc_norm_over_time(values))

    plot_org_vs_corrected(values, images, um_per_pixel, time, time_step, cb_range)

    plt.savefig(fname)
    plt.close("all")


def _make_animation(
    values, images, um_per_pixel, time, cb_range, fname, animation_config
):
    fig, _update = plot_org_vs_corrected(
        values, images, um_per_pixel, time, 0, cb_range
    )

    make_animation(fig, _update, fname, **animation_config)


def _get_avg_data(mps_data, N, fname_data, overwrite):
    if (not overwrite) and os.path.isfile(fname_data):
        fl_avg = np.load(fname_data)
    else:
        fl_avg = mps.analysis.local_averages(mps_data.frames, mps_data.time_stamps, N=N)
        np.save(fname_data, fl_avg)

        print("Local averages found; making the plots ..")
    
    fl_avg = np.swapaxes(np.swapaxes(fl_avg, 0, 1), 0, 2)
    
    return fl_avg


def visualize_fluorescence(f_in, overwrite, overwrite_all, param_list):

    mps_data = mps.MPS(f_in)

    animation_config = get_animation_configuration(param_list[-1], mps_data)
    animate = animation_config.pop("animate")
    time_step = param_list[-1]["time_step"]
    cb_range = (
        param_list[-1]["range_min"],
        param_list[-1]["range_max"],
    )

    print(mps_data.size_x)

    N = min(mps_data.size_x, param_list[-1]["fl_intervals"])
    um_per_pixel = mps_data.info["um_per_pixel"]
    time = mps_data.time_stamps
    images = np.moveaxis(mps_data.frames, 2, 0)

    fname_data = generate_filename(f_in, "plots_spatial", param_list, ".npy")

    print(fname_data)

    fname_png, fname_mp4 = generate_filenames_pngmp4(
        f_in, "plots_spatial", f"orgiginal_vs_corrected_{N}", param_list
    )

    fl_avg = _get_avg_data(mps_data, N, fname_data, overwrite)

    if (not overwrite) and os.path.isfile(fname_png):
        print(f"Image {fname_png} already exists")
    else:
        _plot_at_time_step(
            fl_avg, images, um_per_pixel, time, time_step, cb_range, fname_png
        )
        print(f"Plot at peak done; image saved to {fname_png}")

    if animate:
        if (not overwrite) and os.path.isfile(fname_mp4):
            print(f"Movie {fname_mp4} already exists")
        else:
            print("Making a movie ...")
            _make_animation(
                fl_avg,
                images,
                um_per_pixel,
                time,
                cb_range,
                fname_mp4,
                animation_config,
            )
            print(f"Animation movie produced; movie saved to {fname_mp4}")
