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
from ..utils.iofuns.folder_structure import make_dir_layer_structure
from ..dothemaths.operations import calc_magnitude, normalize_values, calc_norm_over_time
from ..mechanical_analysis.mechanical_analysis import analyze_mechanics


def setup_frame(num_rows, num_cols, dpi, yscale, ymax=None):
    figsize = (14, 12)
    fig, axes = plt.subplots(num_rows, num_cols, \
                             sharex=True, sharey=True, \
                             figsize=figsize, dpi=dpi, squeeze=False)
    
    axes = axes.flatten()
    fig.align_ylabels(axes)

    for axis in axes:
        axis.set_yscale(yscale)

        if ymax is not None:
            axis.set_ylim(0, ymax)

    return axes, fig


def plot_distribution(ax, data, time, min_range, max_range):
    assert len(data.shape)==2, "Error: 2D numpy array expected as input."
     
    values = []
    X, Y = data.shape
    
    for x in range(X):
        for y in range(Y):
            values.append(data[x, y])

    num_bins = 300
    bins = list(np.linspace(min_range, max_range, num_bins))

    ax.set_xlim(min_range, max_range)
    ax.hist(values, bins=bins, color='#28349C')
    #ax.text(1100, 10000, r"$\mu = {:.2f}$".format(np.mean(data)))
    #ax.text(1100, 8500, r"$\sigma = {:.2f}$".format(np.std(data)))
    
    plt.suptitle("Time: {} ms".format(int(time)))


def _get_minmax_range(values, time_step):
    min_range = -np.max(np.abs(values[time_step]))
    max_range = np.max(np.abs(values[time_step]))
    return min_range, max_range

def plot_1Dvalues(values, time_step, yscale, label, time, dpi=None, ymax=None):
    axes, fig = setup_frame(1, 1, dpi, yscale, ymax)
    
    min_range, max_range = _get_minmax_range(values, time_step)

    plot_distribution(axes[0], values[time_step], \
                        time[time_step],
                        min_range, max_range)

    axes[0].set_title(f"Scalar value")

    def update(index):
        plot_distribution(axes[0], values[index], \
                        time[index], \
                        min_range, max_range)

    return fig, update


def plot_2Dvalues(values, time_step, yscale, label, time, dpi=None, ymax=None):
    axes, fig = setup_frame(1, 2, dpi, yscale, ymax)
    
    min_range, max_range = _get_minmax_range(values, time_step)
    
    plot_distribution(axes[0], values[time_step,:,:,0], \
                        time[time_step],
                        min_range, max_range)
    plot_distribution(axes[1], values[time_step,:,:,1], \
                        time[time_step],
                        min_range, max_range)

    axes[0].set_title("x component")
    axes[1].set_title("y component")

    def update(index):
        plot_distribution(axes[0], values[index,:,:,0], \
                    time[index], \
                    min_range, max_range)
        plot_distribution(axes[1], values[index,:,:,1], \
                    time[index], \
                    min_range, max_range)

    return fig, update


def plot_4Dvalues(values, time_step, yscale, label, time, dpi=None, ymax=None):
    axes, fig = setup_frame(2, 2, dpi, yscale, ymax)
    
    min_range, max_range = _get_minmax_range(values, time_step)
    
    plot_distribution(axes[0], values[time_step,:,:,0,0], \
                        time[time_step], \
                        min_range, max_range)
    plot_distribution(axes[1], values[time_step,:,:,0,1], \
                        time[time_step],
                        min_range, max_range)
    plot_distribution(axes[2], values[time_step,:,:,1,0], \
                        time[time_step],
                        min_range, max_range)
    plot_distribution(axes[3], values[time_step,:,:,1,1], \
                        time[time_step],
                        min_range, max_range)

    axes[0].set_title("xx component")
    axes[1].set_title("xy component")
    axes[2].set_title("yx component")
    axes[3].set_title("yy component")

    def update(index):
        plot_distribution(axes[0], values[index,:,:,0,0], \
                        time[index], \
                        min_range, max_range)
        plot_distribution(axes[1], values[index,:,:,0,1], \
                        time[index],
                        min_range, max_range)
        plot_distribution(axes[2], values[index,:,:,1,0], \
                        time[index],
                        min_range, max_range)
        plot_distribution(axes[3], values[index,:,:,1,1], \
                        time[index],
                        min_range, max_range)

    return fig, update


def make_animations(values, yscale, label, fname, time, \
        framerate=None, extension="mp4", dpi=100, ymax=None):
    
    plot_fn = get_plot_fn(values)

    extensions = ["gif", "mp4"]
    msg = "Invalid extension {}. Expected one of {}".format(extension, extensions)
    assert extension in extensions, msg

    fig, update = plot_fn(values, 0, yscale, label, time, dpi=dpi, ymax=ymax)

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
    
    if num_dims == ():
        return plot_1Dvalues
    
    if num_dims == (2,):
        return plot_2Dvalues
    
    if num_dims == (2, 2):
        return plot_4Dvalues
    
    print("Error: shape of {} not recognized.".format(num_dims))


def plot_at_peak(values, yscale, label, time, fname, \
        dpi=300, extension="png"):

    peak = np.argmax(calc_norm_over_time(values))
    plot_fn = get_plot_fn(values)
    
    fig, update = plot_fn(values, peak, yscale, label, time, dpi=dpi)

    ymax = plt.ylim()[1]

    filename = fname + "." + extension
    plt.savefig(filename)
    plt.close('all')

    return ymax


def visualize_distributions(f_in, scaling_factor, type_filter, sigma, animate=False, overwrite=False, save_data=True):
    """

    Make plots for distributions over different quantities - "main function"

    """
    output_folder = make_dir_layer_structure(f_in, \
            os.path.join("mpsmechanics", "distributions"))
    os.makedirs(output_folder, exist_ok=True)
    
    mt_data = mps.MPS(f_in)
    print("Init distributions") 

    source_file = f"analyze_mechanics_{type_filter}_{sigma}"
    source_file = source_file.replace(".", "p")
    
    data = read_prev_layer(f_in, source_file, analyze_mechanics, save_data)
    yscale = "log"
    time = data["time"] 

    for key in ["Green-Lagrange_strain_tensor"]:
        print("Plots for " + key + " ...")

        label = key.capitalize() + "({})".format(data["units"][key])
        label = label.replace("_", " ")

        values = data["all_values"][key]
 
        result_file = f"distribution_{key}_{type_filter}_{sigma}"
        result_file = result_file.replace(".", "p")
        fname = os.path.join(output_folder, result_file)

        ymax = plot_at_peak(values, yscale, label, time, fname)
        
        if animate:
            make_animations(values, yscale, label, fname, \
                    time, framerate=scaling_factor*mt_data.framerate, ymax=ymax)

    print("Distributions plotted, finishing ..")
