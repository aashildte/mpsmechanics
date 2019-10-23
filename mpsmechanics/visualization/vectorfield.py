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

def setup_frame(values, dpi, images, num_rows, num_cols):
    Nx, Ny = images.shape[:2]
    x = np.linspace(0, Nx, values.shape[1])
    y = np.linspace(0, Ny, values.shape[2])

    figsize = (14, 12)
    fig, axes = plt.subplots(num_rows, num_cols, \
                             figsize=figsize, dpi=dpi)

    axes = axes.flatten()
    fig.align_ylabels(axes)

    #for axis in axes:
    #    axis.set_anchor('NW')

    return x, y, axes, fig


def plt_quiver(ax, i, values, x, y, num_arrows, color):
    ax.invert_yaxis()
    
    return ax.quiver(
        y[::num_arrows],
        x[::num_arrows],
        values[i, ::num_arrows, ::num_arrows, 1],
        -values[i, ::num_arrows, ::num_arrows, 0],
        color=color,
        units="xy",
    )


def plt_magnitude(axis, i, scalars, vmin, vmax, cmap, scale):
    """

    Gives a heatmap based on magnitude given in scalars.

    Args:
        axis - defines subplot
        i - time step
        scalars - values; T x X x Y numpy array
        vmin - minimum value possible
        vmax - maximum value possible
        cmap - type of colour map

    """
    if scale == "logscale":
        norm=SymLogNorm(1E-4, vmin=vmin, vmax=vmax)
    else:
        norm=Normalize(vmin=vmin, vmax=vmax)

    return axis.imshow(scalars[i, :, :], vmin=vmin, vmax=vmax, cmap=cmap, norm=norm)

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


def plot_1Dvalues(values, scale, time_step, label, dpi, pixels2um, images):
    x, y, axes, fig = setup_frame(values, dpi, images, 1, 2)
    subplots = []
    
    subplots.append(axes[0].imshow(images[:, :, time_step], cmap=cm.gray))
    subplots.append(plt_magnitude(axes[1], time_step, values[:, :, :, 0], \
                       0, np.max(values), "viridis", scale))
    cb = fig.colorbar(subplots[1], ax=axes[1])
    cb.set_label(label)

    axes[0].set_title("Original images")
    axes[1].set_title("Magnitude")
    
    _set_ax_units(axes[0], images.shape[:2], pixels2um)
    _set_ax_units(axes[1], values.shape[1:3], (images.shape[0]/values.shape[1])*pixels2um)

    def update(index):
        subplots[0].set_array(images[:, :, index])
        subplots[1].set_data(values[index, :, :, 0])

    return fig, update


def plot_2Dvalues(values, scale, time_step, label, num_arrows, dpi, pixels2um, images):

    magnitude = calc_magnitude(values)
    normalized = normalize_values(values)

    x, y, axes, fig = setup_frame(values, dpi, images, 2, 3)

    block_size = len(x) // values.shape[1]
    scale_n = max(np.divide(values.shape[1:3], block_size))

    scale_xy = np.max(np.abs(values))
    
    subplots = []

    subplots.append(axes[0].imshow(images[:, :, time_step], cmap=cm.gray))
    subplots.append(plt_quiver(axes[1], time_step, values, x, y, num_arrows, 'red'))
    subplots.append(plt_quiver(axes[2], time_step, normalized, x, y, num_arrows, 'black'))
    subplots.append(plt_magnitude(axes[3], time_step, magnitude[:, :, :, 0], 0, np.max(magnitude), "viridis", scale))
    cb = fig.colorbar(subplots[3], ax=axes[3])
    cb.set_label(label)

    subplots.append(plt_magnitude(axes[4], time_step, values[:, :, :, 0], -scale_xy, scale_xy, "bwr", scale))
    cb = fig.colorbar(subplots[4], ax=axes[4])
    cb.set_label(label)

    subplots.append(plt_magnitude(axes[5], time_step, values[:, :, :, 1], -scale_xy, scale_xy, "bwr", scale))
    cb = fig.colorbar(subplots[5], ax=axes[5])
    cb.set_label(label)

    axes[0].set_title("Original images")
    axes[1].set_title("Vector field")
    axes[2].set_title("Direction")
    axes[3].set_title("Magnitude")
    axes[4].set_title("Longitudinal (x)")

    axes[5].set_title("Transversal (y)")

    plt.suptitle("Time step {}".format(time_step))
 
    for axis in axes[:3]:
        _set_ax_units(axis, images.shape[:2], pixels2um)

    for axis in axes[3:]:
        _set_ax_units(axis, values.shape[1:3], (images.shape[0]/values.shape[1])*pixels2um)

    def update(index):
        subplots[0].set_array(images[:, :, index])
        subplots[1].set_UVC(values[index, ::num_arrows, ::num_arrows, 1], \
                   -values[index, ::num_arrows, ::num_arrows, 0])
        subplots[2].set_UVC(normalized[index, ::num_arrows, ::num_arrows, 1], \
                   -normalized[index, ::num_arrows, ::num_arrows, 0])
        subplots[3].set_data(magnitude[index, :, :, 0])
        subplots[4].set_data(values[index, :, :, 0])
        subplots[5].set_data(values[index, :, :, 1])

        plt.suptitle("Time step {}".format(index))


    return fig, update



def plot_4Dvalues(values, scale, time_step, label, dpi, pixels2um, images):
    """

    Plots original image, eigenvalues (max.) and four components of
    a tensor value.

    Args:
        values - T x X x Y x 4 numpy value
        time_step - for which time step
        label - description
        dpi - resolution
        pxiels2um - scaling factor for dimensions
        images - original images

    """
    x, y, axes, fig = setup_frame(values, dpi, images, 2, 3)

    block_size = len(x) // values.shape[1]
    scale_n = max(np.divide(values.shape[1:3], block_size))

    scale_xy = np.max(np.abs(values))

    sg_values = np.linalg.norm(values, axis=(3, 4))

    subplots = [axes[0].imshow(images[:, :, time_step], cmap=cm.gray),
                plt_magnitude(axes[1], time_step, values[:,:,:,0,0], \
                    -scale_xy, scale_xy, "bwr", scale),
                plt_magnitude(axes[2], time_step, values[:,:,:,0,1], \
                    -scale_xy, scale_xy, "bwr", scale),
                plt_magnitude(axes[3], time_step, sg_values, \
                        0, scale_xy, "viridis", scale),
                plt_magnitude(axes[4], time_step, values[:,:,:,1,0], \
                    -scale_xy, scale_xy, "bwr", scale),
                plt_magnitude(axes[5], time_step, values[:,:,:,1,1], \
                    -scale_xy, scale_xy, "bwr", scale)]

    for i in [1, 2, 4, 5]:
        cb = fig.colorbar(subplots[i], ax=axes[i])
        cb.set_label(label)

    axes[0].set_title("Original image")
    axes[1].set_title(r"$u_x$")
    axes[2].set_title(r"$u_y$")
    axes[3].set_title("Largest singular value")
    axes[4].set_title(r"$v_x$")
    axes[5].set_title(r"$v_y$")

    plt.suptitle("Time step {}".format(time_step))

    _set_ax_units(axes[0], images.shape[:2], pixels2um)
    for axis in axes[1:]:
        _set_ax_units(axis, values.shape[1:3], (images.shape[0]/values.shape[1])*pixels2um)

    def update(index):
        subplots[0].set_array(images[:, :, index])
        subplots[1].set_data(values[index, :, :, 0, 0])
        subplots[2].set_data(values[index, :, :, 1, 0])
        subplots[3].set_data(sg_values[index, :, :])
        subplots[4].set_data(values[index, :, :, 0, 1])
        subplots[5].set_data(values[index, :, :, 1, 1])
    
        plt.suptitle("Time step {}".format(index))

    return fig, update


def animate_vectorfield(values, scale, label, pixels2um, images, fname="animation", \
        framerate=None, extension="mp4", dpi=300, dx=3):

    extensions = ["gif", "mp4"]
    msg = "Invalid extension {}. Expected one of {}".format(extension, extensions)
    assert extension in extensions, msg

    num_dims = values.shape[3:]

    if num_dims == (1,):
        fig, update = plot_1Dvalues(values, scale, 0, label, dpi, pixels2um, images)
    elif num_dims == (2,):
        fig, update = plot_2Dvalues(values, scale, 0, label, dx, dpi, pixels2um, images)
    elif num_dims == (2, 2):
        fig, update = plot_4Dvalues(values, scale, 0, label, dpi, pixels2um, images)
    else:
        print("Error: shape of {} not recognized.".format(num_dims))
        return

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


def plot_at_peak(values, scale, label, pixels2um, images, fname="vector_fields", \
        extension="png", dpi=600, num_arrows=3):

    num_dims = values.shape[3:]

    if num_dims == (1,):
        peak = np.argmax(calc_norm_over_time(values))
        plot_1Dvalues(values, scale, peak, label, dpi, pixels2um, images)

    elif num_dims == (2,):
        peak = np.argmax(calc_norm_over_time(values))
        plot_2Dvalues(values, scale, peak, label, num_arrows, dpi, pixels2um, images)

    elif num_dims == (2, 2):
        peak = np.argmax(calc_norm_over_time(np.linalg.norm(values, \
                axis=(3, 4))[:, :, :, None]))
        plot_4Dvalues(values, scale, peak, label, dpi, pixels2um, images)

    else:
        print("Error: shape of {} not recognized.".format(num_dims))
        return

    filename = fname + "." + extension
    plt.savefig(filename)


def visualize_vectorfield(f_in, framerate_scale=1, save_data=True):
    """

    Visualize fields - "main function"

    """

    mt_data = mps.MPS(f_in)
    pixels2um = mt_data.info["um_per_pixel"]

    output_folder = make_dir_layer_structure(f_in, \
            os.path.join("mpsmechanics", "visualize_vectorfield"))
    make_dir_structure(output_folder)

    data = read_prev_layer(f_in, "analyze_mechanics", analyze_mechanics, save_data)
    print(data["all_values"].keys())
    for key in data["all_values"].keys():
        for scale in ["logscale", "linear"]:
            print("Plots for " + key + " ...")
            label = key.capitalize() + "({})".format(data["units"][key])
            plot_at_peak(data["all_values"][key], scale, label, pixels2um, mt_data.data.frames, \
                     fname=os.path.join(output_folder, "vectorfield_" + scale + "_" + key))

            animate_vectorfield(data["all_values"][key], scale, label, pixels2um, \
                            mt_data.data.frames, framerate=framerate_scale*mt_data.framerate, \
                            fname=os.path.join(output_folder, "vectorfield_" + scale + "_" + key))

    print("Visualization done, finishing ..")
