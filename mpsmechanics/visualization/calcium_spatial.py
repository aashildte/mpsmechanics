"""

Åshild Telle / Simula Research Laboratory / 2019

"""


import numpy as np
import matplotlib as plt
import mps
import mpsmechanics as mm
import glob
import os
import sys
from argparse import ArgumentParser


def _make_1d_plot_pretty(fig, axes, subplots, subtitles, metadata):
    label = metadata["label"]
    pixels2um = metadata["pixels2um"]
    blocksize = metadata["blocksize"]

    fig.colorbar(subplots[1], ax=axes[1]).set_label(label)

    for (axis, subtitle) in zip(axes, subtitles):
        axis.set_title(subtitle)

    _set_ax_units(axes[0], pixels2um, 0)
    _set_ax_units(axes[1], blocksize * pixels2um, blocksize // 2)


def plot_1d_values(images, values, time, time_step, metadata):
    """

    Plots original image and magnitude.

    Args:
        images : original images : T x X x Y numpy array
        values : T x X x Y numpy array
        time - corresponding time units
        time_step - integer value; time step of interest, typically peak or 0
        metadata - dictionary with information about labels, units

    """

    all_components = [images, values]
    subtitles = [
        "Original image",
        "Pixel intensity, relative to baseline",
    ]

    axes, fig, subplots = _init_subplots_1d(
        all_components, time_step
    )
    _make_1d_plot_pretty(fig, axes, subplots, subtitles, metadata)
    plt.suptitle("Time: {} ms".format(int(time[time_step])))

    def _update(index):
        subplots[0].set_array(images[index])
        subplots[1].set_data(values[index])
        plt.suptitle("Time: {} ms".format(int(time[index])))

    return fig, _update


def _plot_at_peak(avg, time, metadata, fname):

    values = spatial_data["derived_quantity"]

    peak = np.argmax(calc_norm_over_time(values))
    plot_fn = get_plot_fun(
        values, [plot_1d_values, plot_2d_values, plot_2x2d_values]
    )
    plot_fn(spatial_data, time, peak, metadata)

    plt.savefig(fname)
    plt.close("all")


def visualize_calcium(
    input_file, animate=False, overwrite=False, framerate_scale=0.2
):

    data = mps.MPS(input_file)
    N = data.size_x

    output_folder = os.path.join(input_file[:-4], "plots_spatial")
    fout = os.path.join(output_folder, f"corrected_{N}.npy")

    if (not overwrite) and os.path.isfile(fout):
        return

    os.makedirs(output_folder, exist_ok=True)

    avg = mps.analysis.local_averages(
        data.frames, data.time_stamps, N=N
    )
    np.save(fout, avg)

    print(
        f"Local averages found for {input_file}; making the plots .."
    )

    avg = np.swapaxes(np.swapaxes(avg, 0, 1), 0, 2)
    avg = avg[:, :, :, None]

    pixels2um = data.info["um_per_pixel"]
    fname = os.path.join(output_folder, "original_vs_corrected")
    _plot_at_peak(avg, pixels2um, data.frames, fname)

    if animate:
        print("Making a movie ...")
        _make_animation(
            avg,
            pixels2um,
            data.frames,
            fname,
            framerate=framerate_scale * data.framerate,
        )
