"""

Åshild Telle / Simula Research Laboratory / 2019

"""


import numpy as np
import matplotlib as plt
import mps
import glob
import os
import sys
from argparse import ArgumentParser

from mpsmechanics.dothemaths.operations import calc_norm_over_time


def _init_subplots(images, values, time, time_step, metadata):
    """

    Plots original image and magnitude.

    Args:
        images - original Cyan images : T x X x Y numpy array
        values - corrected trace : T x X x Y numpy array
        time - corresponding time units
        time_step - integer value; time step of interest, typically peak or 0
        metadata - dictionary with information about labels, units

    """

    axes, fig = setup_frame(1, 2, False, False)
    
    subplots = []

    subplots.append(axes[0].imshow(images[time_step], cmap="gray"))
    subplots.append(make_heatmap_plot(axes[1], values[time_step], \
                       0, np.max(magnitude), "viridis"))
    
    axes[0].set_title("Original values")
    axes[1].set_title("Corrected trace")

    plt.suptitle("Time: {} ms".format(int(time[time_step])))

    def _update(index):
        subplots[0].set_array(images[index])
        subplots[1].set_data(values[index])
        plt.suptitle("Time: {} ms".format(int(time[index])))

    return fig, _update


def _plot_at_peak(images, values, time, metadata, fname):

    peak = np.argmax(calc_norm_over_time(values))
    _init_subplots(images, values, time, peak, metadata)

    plt.savefig(fname)
    plt.close('all')


def _make_animation(images, values, time, metadata, fname, animation_config):
    fig, _update = _init_subplots(images, values, time, 0, metadata)

    make_animation(fig, _update, fname, **animation_config)


def visualize_calcium_spatial(input_file, animate=False, overwrite=False, \
        magnitude_scale="linear", framerate_scale=0.2):
    
    data = mps.MPS(input_file)    
    N = data.size_x // 12
    print(f"N: {N}")

    output_folder = os.path.join(input_file[:-4], "plots_spatial")
    fout = os.path.join(output_folder, f"corrected_{N}.npy")
    
    if (not overwrite) and os.path.isfile(fout):
        return
    
    os.makedirs(output_folder, exist_ok=True)

    avg = mps.analysis.local_averages(data.frames, data.time_stamps, N=N)
    np.save(fout, avg)

    print(f"Local averages found for {input_file}; making the plots ..")

    avg = np.swapaxes(np.swapaxes(avg, 0, 1), 0, 2)
    avg = avg[:,:,:,None]
    
    pixels2um = data.info["um_per_pixel"]
    label = "Pixel intensity, relative to baseline"
    fname = os.path.join(output_folder, "original_vs_corrected")

    _plot_at_peak(avg, magnitude_scale, label, pixels2um, data.frames, fname)

    if animate:
        print("Making a movie ...")
        _make_animation(avg, magnitude_scale, label, pixels2um, \
             data.frames, fname, framerate=framerate_scale*data.framerate)
