"""

Åshild Telle / Simula Research Laboratory / 2019

"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation
from matplotlib.colors import SymLogNorm, Normalize
import multiprocessing as mp

import mps

from ..utils.data_layer import read_prev_layer, generate_filename
from ..utils.folder_structure import make_dir_layer_structure
from ..dothemaths.operations import calc_magnitude, normalize_values, calc_norm_over_time
from ..mechanical_analysis.mechanical_analysis import analyze_mechanics
from .animation_funs import make_animation, get_animation_configuration
from .setup_plots import setup_frame, get_plot_fun, make_pretty_label, load_input_data

def setup_frame(values, dpi, images, num_rows, num_cols):
    Nx, Ny = images.shape[:2]
    x = np.linspace(0, Nx, values.shape[1])
    y = np.linspace(0, Ny, values.shape[2])

    figsize = (14, 12)
    fig, axes = plt.subplots(num_rows, num_cols, \
                             figsize=figsize, dpi=dpi, squeeze=False)
    
    axes = axes.flatten()
    fig.align_ylabels(axes)

    return x, y, axes, fig


def plt_quiver(ax, i, values, x, y, num_arrows, color, scale):
    ax.invert_yaxis()
    
    return ax.quiver(
        y[::num_arrows],
        x[::num_arrows],
        values[i, ::num_arrows, ::num_arrows, 1],
        -values[i, ::num_arrows, ::num_arrows, 0],
        color=color,
        units="xy",
        headwidth=6,
        scale=scale,
    )


def _set_ax_units(axis, shape, scale):
    
    axis.set_aspect("equal")
    num_yticks = 8
    num_xticks = 4

    yticks = [int(shape[0]*i/num_yticks) for i in range(num_yticks)]
    ylabels = [int(scale*shape[0]*i/num_yticks) for i in range(num_yticks)]

    xticks = [int(shape[1]*i/num_xticks) for i in range(num_xticks)]
    xlabels = [int(scale*shape[1]*i/num_xticks) for i in range(num_xticks)]

    axis.set_yticks(yticks)
    axis.set_xticks(xticks)
    axis.set_yticklabels(ylabels)
    axis.set_xticklabels(xlabels)

    axis.set_xlabel(r"$\mu m$")
    axis.set_ylabel(r"$\mu m$")


def _find_arrow_scaling(values, num_arrows):
    return 1.5/(num_arrows)*np.mean(np.abs(values))

def plot_vectorfield(spatial_data, time, time_step, metadata):
    
    values = spatial_data["derived_quantity"]
    images = spatial_data["images"]
    label = metadata["label"]
    pixels2um = metadata["pixels2um"]

    dpi=300

    x, y, axes, fig = setup_frame(values, dpi, images, 1, 1)

    block_size = len(x) // values.shape[1]
    scale_n = max(np.divide(values.shape[1:3], block_size))
    num_arrows = 3

    scale_xy = np.max(np.abs(values))
    scale_arrows = _find_arrow_scaling(values, num_arrows)
    
    Q1 = axes[0].imshow(images[:, :, time_step], cmap=cm.gray) 
    Q2 = plt_quiver(axes[0], time_step, values, x, y, num_arrows, 'red', scale_arrows)

    plt.suptitle("Time: {} ms".format(int(time[time_step])))
 
    _set_ax_units(axes[0], images.shape[:2], pixels2um)

    def update(index):
        Q1.set_array(images[:, :, index])
        Q2.set_UVC(values[index, ::num_arrows, ::num_arrows, 1], \
                   values[index, ::num_arrows, ::num_arrows, 0])

        plt.suptitle("Time: {} ms".format(int(time[index])))

    return fig, update


def _plot_at_peak(spatial_data, time, metadata, fname):
 
    values = spatial_data["derived_quantity"]
    peak = np.argmax(calc_norm_over_time(values))
         
    plot_vectorfield(spatial_data, time, peak, metadata)

    filename = fname + ".png"
    plt.savefig(filename)
    plt.close('all')
    

def _make_animation(values, time, label, pixels2um, images, fname, \
                        framerate=None, extension="mp4", dpi=300, num_arrows=3):
    
    fig, update = plot_vectorfield(values, time, 0, label, num_arrows, dpi, pixels2um, images)

    if extension == "mp4":
        Writer = animation.writers["ffmpeg"]
    else:
        Writer = animation.writers["imagemagick"]
    writer = Writer(fps=framerate)

    N = values.shape[0]
    anim = animation.FuncAnimation(fig, update, N)

    fname = os.path.splitext(fname)[0]
    anim.save("{}.{}".format(fname, extension), writer=writer)
    plt.close('all')
    


def _make_vectorfield_plots(values, time, key, label, output_folder, pixels2um, \
        images, framerate, animate, fname):
     
    plot_vectorfield_at_peak(values, time, label, pixels2um, \
                    images, fname)

    if animate:
        print("Making a movie ..")
        animate_vectorfield(values, time, label, pixels2um, \
                    images, fname, framerate=framerate)
    
    
def visualize_vectorfield(f_in, overwrite, overwrite_all, param_list):
    """

    "main function"

    Args:
        f_in - BF / nd2 file
        overwrite - make plots again if they don't exist or not
        overwrite_all - recalculate data from previous layers or not
        param_list - list of lists; 3 sublists. First 2 are passed to
            previous layers if data needs to be recalculated; last gives
            parameters for this script.

    """

    print("Parameters visualize vectorfield:")

    for key in param_list[2].keys():
        print(" * {}: {}".format(key, param_list[2][key]))

    mps_data, mc_data = load_input_data(f_in, param_list, overwrite_all)
    animation_config = get_animation_configuration(param_list[2], mps_data)
    animate = animation_config.pop("animate")

    images = np.moveaxis(mps_data.frames, 2, 0)
    time = mc_data["time"]

    for key in mc_data["all_values"].keys():
        if mc_data["all_values"][key].shape[3:] != (2,):
            continue

        print("Plots for " + key + " ...")

        fname = generate_filename(f_in, \
                                  f"decomposition_{key}", \
                                  param_list[:2],
                                  "",        # mp3 or png
                                  subfolder="visualize_mechanics")

        values = mc_data["all_values"][key]

        metadata = {"label" : make_pretty_label(key, mc_data["unit"][key]),
                    "pixels2um" : mps_data.info["um_per_pixel"],
                    "blocksize" : images.shape[1] // values.shape[1]}

        spatial_data = {"images" : images,
                        "derived_quantity" : values}

        if overwrite or (not os.path.isfile(fname + ".png")):
            print("Spatial plots ..")
            _plot_at_peak(spatial_data, time, metadata, fname)

        if animate and (overwrite or (not os.path.isfile(fname + ".mp4"))):
            print("Making a movie ..")
            _make_animation(spatial_data, time, metadata, fname, \
                    animation_config)

    print("Visualization done, finishing ...")
