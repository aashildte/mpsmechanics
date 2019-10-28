"""

Åshild Telle / Simula Research Laboratory / 2019

"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation
from matplotlib.colors import SymLogNorm, Normalize

import mps

from ..utils.iofuns.data_layer import read_prev_layer
from ..utils.iofuns.folder_structure import make_dir_structure, \
        make_dir_layer_structure
from ..dothemaths.operations import calc_magnitude, normalize_values, calc_norm_over_time
from ..mechanical_analysis.mechanical_analysis import analyze_mechanics


def setup_frame(num_rows, num_cols, dpi, yscale):
    figsize = (14, 12)
    fig, axes = plt.subplots(num_rows, num_cols, \
                             sharex=True, sharey=True, \
                             figsize=figsize, dpi=dpi, squeeze=False)
    
    axes = axes.flatten()
    fig.align_ylabels(axes)

    for axis in axes:
        axis.set_yscale(yscale)

    return axes, fig

def plot_distribution(ax, data, disp_filter):

    assert len(data.shape)==2, "Error: 2D numpy array expected as input."
 
    values = []

    X, Y = data.shape
    
    for x in range(X):
        for y in range(Y):
            if disp_filter[x, y]:
                values.append(data[x, y])

    bins = list(np.linspace(np.min(values), np.max(values), 20))
    #bins.sort()

    ax.hist(values, bins=bins, color='#28349C')
    #ax.text(1100, 10000, r"$\mu = {:.2f}$".format(np.mean(data)))
    #ax.text(1100, 8500, r"$\sigma = {:.2f}$".format(np.std(data)))
    #ax.text(1100, 7000, r"$n = {}$".format(np.sum(disp_filter)))

def plot_1Dvalues(values, time_step, disp_filter, yscale, label, dpi=None):
    axes, fig = setup_frame(1, 1, dpi, yscale)
    
    plot_distribution(axes[0], values[time_step,:,:,0], \
                        disp_filter[time_step])

    axes[0].set_title(f"Scalar value")

    def update(index):
        plot_distribution(axes[0], data[index,:,:,0], \
                        disp_filter[index])

    return fig, update


def plot_2Dvalues(values, time_step, disp_filter, yscale, label, dpi=None):
    axes, fig = setup_frame(1, 2, dpi, yscale)
    
    plot_distribution(axes[0], values[time_step,:,:,0], \
                        disp_filter[time_step])
    plot_distribution(axes[1], values[time_step,:,:,1], \
                        disp_filter[time_step])

    axes[0].set_title("x component")
    axes[1].set_title("y component")

    def update(index):
        plot_distribution(axes[0], data[index,:,:,0], \
                    disp_filter[index])
        plot_distribution(axes[1], data[index,:,:,1], \
                    disp_filter[index])

    return fig, update

def plot_4Dvalues(values, time_step, disp_filter, yscale, label, dpi=None):
    axes, fig = setup_frame(2, 2, dpi, yscale)
    
    plot_distribution(axes[0], values[time_step,:,:,0,0], \
                        disp_filter[time_step])
    plot_distribution(axes[1], values[time_step,:,:,0,1], \
                        disp_filter[time_step])
    plot_distribution(axes[2], values[time_step,:,:,1,0], \
                        disp_filter[time_step])
    plot_distribution(axes[3], values[time_step,:,:,1,1], \
                        disp_filter[time_step])

    axes[0].set_title("xx component")
    axes[1].set_title("xy component")
    axes[2].set_title("yx component")
    axes[3].set_title("yy component")

    def update(index):
        plot_distribution(axes[0], data[index,:,:,0], \
                    disp_filter[index])
        plot_distribution(axes[1], data[index,:,:,1], \
                    disp_filter[index])

    return fig, update


def animate_vectorfield(values, scale_magnitude, label, pixels2um, images, fname, \
        framerate=None, extension="mp4", dpi=300, num_arrows=3, scale_arrows=None):
    # TODO move to separate file?
    plot_fn = get_plot_fn(values)

    extensions = ["gif", "mp4"]
    msg = "Invalid extension {}. Expected one of {}".format(extension, extensions)
    assert extension in extensions, msg

    fig, update, _ = plot_fn(values, scale_magnitude, 0, label, dpi, \
                        pixels2um, images, scale_arrows=scale_arrows)

    # Set up formatting for the movie files
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


def get_plot_fn(values):
    num_dims = values.shape[3:]
    
    if num_dims == (1,):
        return plot_1Dvalues
    
    if num_dims == (2,):
        return plot_2Dvalues
    
    if num_dims == (2, 2):
        return plot_4Dvalues
    
    print("Error: shape of {} not recognized.".format(num_dims))


def plot_at_peak(values, disp_filter, yscale, label, fname, \
        dpi=300, extension="png"):

    peak = np.argmax(calc_norm_over_time(values))
    plot_fn = get_plot_fn(values)
    
    plot_fn(values, peak, disp_filter, yscale, label, dpi=dpi)

    filename = fname + "." + extension
    plt.savefig(filename)
    plt.close('all')


def visualize_distributions(f_in, framerate_scale, save_data=True):
    """

    Make plots for distributions over different quantities - "main function"

    """
    output_folder = make_dir_layer_structure(f_in, \
            os.path.join("mpsmechanics", "distributions"))
    make_dir_structure(output_folder)

    data = read_prev_layer(f_in, "analyze_mechanics", analyze_mechanics, save_data)
    
    for key in data["all_values"].keys():
        print("Plots for " + key + " ...")

        label = key.capitalize() + "({})".format(data["units"][key])
    
        for yscale in ["linear", "log"]:    
            fname = os.path.join(output_folder, f"distribution_{yscale}_{key}")
            plot_at_peak(data["all_values"][key], data["filters"][key], \
                    yscale, label, fname)

            #make_animations(data["all_values"][key], label, pixels2um, fname, \
            #        framerate=framerate_scale*mt_data.framerate)

    print("Distributions plotted, finishing ..")
