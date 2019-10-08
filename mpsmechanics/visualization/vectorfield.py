"""

Åshild Telle / Simula Research Laboratory / 2019

"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation

import mps

from ..utils.iofuns.data_layer import read_prev_layer
from ..utils.iofuns.folder_structure import make_dir_structure, \
        make_dir_layer_structure
from ..dothemaths.operations import calc_magnitude, normalize_values, calc_norm_over_time
from ..mechanical_analysis.mechanical_analysis import analyze_mechanics

def setup_frame(vectors, dpi, images, num_rows, num_cols):
    Nx, Ny = images.shape[:2]
    x = np.linspace(0, Nx, vectors.shape[1])
    y = np.linspace(0, Ny, vectors.shape[2])

    figsize = (14, 10)
    fig, axes = plt.subplots(num_rows, num_cols, \
                             figsize=figsize, dpi=dpi)
    axes.flatten()

    return x, y, axes, fig


def plt_quiver(ax, i, vectors, x, y, num_arrows, scale, color):
    ax.invert_yaxis()
    return ax.quiver(
        y[::num_arrows],
        x[::num_arrows],
        vectors[i, ::num_arrows, ::num_arrows, 1],
        -vectors[i, ::num_arrows, ::num_arrows, 0],
        color=color,
        units="xy",
        scale=scale,
    )


def plt_magnitude(axis, i, scalars, vmin, vmax, cmap):
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
    return axis.imshow(scalars[i, :, :], vmin=vmin, vmax=vmax, cmap=cmap)


def _set_axes(axes, x, images, pixels2um):
    scale_ax = [1, len(x)/images.shape[0]]

    for axis in axes:
        axes.set_aspect("equal")

        axes.set_yticks(np.linspace(0, scale_ax[i]*images.shape[0]-1, 8))
        axes.set_yticklabels(map(int, np.linspace(0, pixels2um*(images.shape[0]-1), 8)))
        axes.set_xticks(np.linspace(0, scale_ax[i]*images.shape[1]-1, 5))
        axes.set_xticklabels(map(int, np.linspace(0, pixels2um*(images.shape[1]-1), 5)))
        axes.set_xlabel("$\mu m$")
        axes.set_ylabel("$\mu m$")


def plot_1Dvalues(scalars, time_step, label, num_arrows, dpi, pixels2um, scale, images):
    x, y, axes, fig = setup_frame(vectors, dpi, images, 2, 1)
    subplots = []

    block_size = len(x) // vectors.shape[1]
    scale_n = max(np.divide(vectors.shape[1:3], block_size))

    scale_xy = np.max(np.abs(vectors))

    subplots.append(axes[0].imshow(images[:,:,time_step], cmap=cm.gray))
    subplots.append(plt_magnitude(axes[1], time_step, scalars[:, :, :, 0], \
                       0, np.max(magnitude), "viridis"))
    cb = fig.colorbar(subplots[1], ax=axes[1, 0])
    cb.set_label(label)

    axes[0].set_title("Original images")
    axes[1].set_title("Magnitude")

    _set_axes(axes, x, images, pixels2um)

    def update(index):
        subplots[0].set_array(images[:, :, index])
        subplots[1].set_data(scalars[index, :, :, 0])

    return fig, update


def plot_2Dvalues(vectors, time_step, label, num_arrows, dpi, pixels2um, scale, images):

    magnitude = calc_magnitude(vectors)
    normalized = normalize_values(vectors)

    x, y, axes, fig = setup_frame(vectors, dpi, images)

    block_size = len(x) // vectors.shape[1]
    scale_n = max(np.divide(vectors.shape[1:3], block_size))

    scale_xy = np.max(np.abs(vectors))

    Q0 = axes[0].imshow(images[:,:,time_step], cmap=cm.gray)
    Q1 = plt_quiver(axes[1], time_step, vectors, x, y, num_arrows, scale*scale_n, 'red')
    Q2 = plt_quiver(axes[2], time_step, normalized, x, y, num_arrows, 1/25, 'black')
    Q3 = plt_magnitude(axes[3], time_step, magnitude[:,:,:,0], 0, np.max(magnitude), "viridis")
    cb = fig.colorbar(Q3, ax=axes[3])
    cb.set_label(label)

    Q4 = plt_magnitude(axes[4], time_step, vectors[:,:,:,0], -scale_xy, scale_xy, "bwr")
    cb = fig.colorbar(Q4, ax=axes[4])
    cb.set_label(label)

    Q5 = plt_magnitude(axes[5], time_step, vectors[:,:,:,1], -scale_xy, scale_xy, "bwr")
    cb = fig.colorbar(Q5, ax=axes[5])
    cb.set_label(label)

    axes[0].set_title("Original images")
    axes[1].set_title("Vector field")
    axes[2].set_title("Direction")
    axes[3].set_title("Magnitude")
    axes[4].set_title("Longitudinal (x)")

    axes[5].set_title("Transversal (y)")

    plt.suptitle("Time step {time_step}".format())
 
    _set_axes(axes, x, images, pixels2um)

    def update(index):
        Q0.set_array(images[:, :, index])
        Q1.set_UVC(vectors[index, ::num_arrows, ::num_arrows, 1], \
                   -vectors[index, ::num_arrows, ::num_arrows, 0])
        Q2.set_UVC(normalized[index, ::num_arrows, ::num_arrows, 1], \
                   -normalized[index, ::num_arrows, ::num_arrows, 0])
        Q3.set_data(magnitude[index, :, :, 0])
        Q4.set_data(vectors[index, :, :, 0])
        Q5.set_data(vectors[index, :, :, 1])

        plt.suptitle("Time step {index}".format())


    return fig, update



def plot_4Dvalues(vectors, time_step, label, dpi, pixels2um, scale, images):
    """

    Plots original image, eigenvalues (max.) and four components of
    a tensor value.

    Args:
        vectors - T x X x Y x 4 numpy value
        time_step - for which time step
        label - description
        dpi - resolution
        pxiels2um - scaling factor for dimensions
        scale - scaling factor for arrows
        images - original images

    """
    x, y, axes, fig = setup_frame(vectors, dpi, images)

    block_size = len(x) // vectors.shape[1]
    scale_n = max(np.divide(vectors.shape[1:3], block_size))

    scale_xy = np.max(np.abs(vectors))

    sg_values = np.linalg.norm(vectors, axis=(3, 4))

    subplots = [axes[0, 0].imshow(images[:, :, time_step], cmap=cm.gray)]
    subplots.append(plt_magnitude(axes[1,0], time_step, sg_values, 0, scale_xy, "viridis"))

    cb = fig.colorbar(subplots[1], ax=axes[1, 0])
    cb.set_label(label)

    for i in range(2):
        for j in range(2):
            subplots.append(plt_magnitude(axes[i, j+1], time_step, vectors[:,:,:,i,j], \
                    -scale_xy, scale_xy, "bwr"))
            cb = fig.colorbar(subplots[-1], ax=axes[i, j+1])
            cb.set_label(label)

    axes[0].set_title("Original image")
    axes[1].set_title(r"$u_x$")
    axes[2].set_title(r"$u_y$")
    axes[3].set_title("Largest singular value")
    axes[4].set_title(r"$v_x$")
    axes[5].set_title(r"$v_y$")

    plt.suptitle("Time step {time_step}".format())

    _set_axes(axes, x, images, pixels2um)

    def update(index):
        subplots[0].set_array(images[:, :, index])
        subplots[1].set_data(vectors[index, :, :, 0, 0])
        subplots[2].set_data(vectors[index, :, :, 1, 0])
        subplots[3].set_data(sg_values[index, :, :])
        subplots[4].set_data(vectors[index, :, :, 0, 1])
        subplots[5].set_data(vectors[index, :, :, 1, 1])
    
        plt.suptitle("Time step {index}".format())

    return fig, update


def animate_vectorfield(vectors, label, pixels2um, images, fname="animation", \
        framerate=None, extension="mp4", dpi=300, dx=3, scale=1):

    extensions = ["gif", "mp4"]
    msg = "Invalid extension {extension}. Expected one of {extensions}".format()
    assert extension in extensions, msg

    num_dims = vectors.shape[3:]

    if num_dims == (1,):
        fig, update = plot_1Dvalues(vectors, 0, label, dpi, pixels2um, scale, images)
    elif num_dims == (2,):
        fig, update = plot_1Dvalues(vectors, 0, label, dx, dpi, pixels2um, scale, images)
    elif num_dims == (2, 2):
        fig, update = plot_4Dvalues(vectors, 0, label, dpi, pixels2um, scale, images)
    else:
        print("Error: shape of {num_dims} not recognized.".format())
        return

    # Set up formatting for the movie files
    if extension == "mp4":
        Writer = animation.writers["ffmpeg"]
    else:
        Writer = animation.writers["imagemagick"]
    writer = Writer(fps=framerate)

    N = vectors.shape[0]
    anim = animation.FuncAnimation(fig, update, N)

    fname = os.path.splitext(fname)[0]
    anim.save("{fname}.{extension}".format(), writer=writer)


def plot_at_peak(vectors, label, pixels2um, images, fname="vector_fields", \
        extension="png", dpi=600, num_arrows=3, scale=1):

    num_dims = vectors.shape[3:]

    if num_dims == (1,):
        peak = np.argmax(vectors)
        plot_1Dvalues(vectors, peak, label, num_arrows, dpi, pixels2um, scale, images)

    elif num_dims == (2,):
        peak = np.argmax(calc_norm_over_time(vectors))
        plot_2Dvalues(vectors, peak, label, num_arrows, dpi, pixels2um, scale, images)

    elif num_dims == (2, 2):
        peak = np.argmax(calc_norm_over_time(np.linalg.norm(vectors, \
                axis=(3, 4))[:, :, :, None]))
        plot_4Dvalues(vectors, peak, label, dpi, pixels2um, scale, images)

    else:
        print("Error: shape of {num_dims} not recognized.".format())
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

    scales = {"displacement" : 6E-5, \
                  "principal strain" : 6E-5, \
                  "velocity" : 1E-6,
                  "xmotion" : None,
                  "prevalence" : None}

    for key in ["displacement", "principal strain", "velocity"]:

        print("Plots for " + key + " ...")
        label = key.capitalize() + "({})".format(data["units"][key])
        plot_at_peak(data["all_values"][key], label, pixels2um, mt_data.data.frames, \
                     fname=os.path.join(output_folder, "vectorfield_" + key),
                     scale=scales[key])

        animate_vectorfield(data["all_values"][key], label, pixels2um, \
                            mt_data.data.frames, framerate=framerate_scale*mt_data.framerate, \
                            fname=os.path.join(output_folder, "vectorfield_" + key),\
                            scale=scales[key])

    print("Visualization done, finishing ..")
