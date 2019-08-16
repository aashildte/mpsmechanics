
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
from matplotlib import animation

import mps

from ..utils.iofuns.data_layer import read_prev_layer
from ..utils.iofuns.folder_structure import make_dir_structure, \
        make_dir_layer_structure
from ..dothemaths.operations import calc_magnitude, normalize_values, calc_norm_over_time
from ..mechanical_analysis.mechanical_analysis import analyze_mechanics
from ..pillar_tracking.pillar_tracking import track_pillars

def setup_frame(vectors, pixels2um, dpi, images):

    Nx, Ny = images.shape[:2]

    x = np.linspace(0, Nx, vectors.shape[1])
    y = np.linspace(0, Ny, vectors.shape[2])

    block_size = Nx // vectors.shape[1]

    figsize = (14, 10)
    
    fig, axs = plt.subplots(2, 3, figsize=figsize, dpi=dpi)
    return x, y, axs, fig


def plt_quiver(ax, i, vectors, x, y, dx, scale, color):
    ax.invert_yaxis()
    return ax.quiver(
        y[::dx],
        x[::dx],
        vectors[i, ::dx, ::dx, 1],
        -vectors[i, ::dx, ::dx, 0],
        color=color,
        units="xy",
        scale=scale,
    )


def plt_magnitude(ax, i, scalars, vmin, vmax, cmap, linscale):
    return ax.imshow(scalars[i, :, :], norm=colors.SymLogNorm(linscale, vmin=vmin, vmax=vmax), cmap=cmap)


def plot2Dvalues(vectors, time_step, label, unit, dx, dpi, pixels2um, scale, images, linscale):

    magnitude = calc_magnitude(vectors)
    normalized = normalize_values(vectors)
    
    x, y, axs, fig = setup_frame(vectors, pixels2um, dpi, images)

    block_size = len(x) // vectors.shape[1]
    scale_n = max(np.divide(vectors.shape[1:3], block_size))

    scale_xy = np.max(np.abs(vectors))

    Q0 = axs[0, 0].imshow(images[:,:,time_step], cmap=cm.gray)
    Q1 = plt_quiver(axs[0, 1], time_step, vectors, x, y, dx, scale*scale_n, 'red')
    Q2 = plt_quiver(axs[0, 2], time_step, normalized, x, y, dx, 1/25, 'black')
    Q3 = plt_magnitude(axs[1, 0], time_step, magnitude[:,:,:,0], 0, np.max(magnitude), "viridis", linscale)
    cb = fig.colorbar(Q3, ax=axs[1, 0])
    cb.set_label(label.capitalize() + " (" + unit + ")")
    
    Q4 = plt_magnitude(axs[1, 1], time_step, vectors[:,:,:,0], -scale_xy, scale_xy, "bwr", linscale)
    cb = fig.colorbar(Q4, ax=axs[1, 1])
    cb.set_label(label.capitalize() + " (" + unit + ")")
    
    Q5 = plt_magnitude(axs[1, 2], time_step, vectors[:,:,:,1], -scale_xy, scale_xy, "bwr", linscale)
    cb = fig.colorbar(Q5, ax=axs[1, 2])
    cb.set_label(label.capitalize() + " (" + unit + ")")
    
    axs[0, 0].set_title("Original images")
    axs[0, 1].set_title("Vector field")
    axs[0, 2].set_title("Direction")
    axs[1, 0].set_title("Magnitude")
    axs[1, 1].set_title("Longitudinal (x)")
    axs[1, 2].set_title("Transversal (y)")
    plt.suptitle(f"Time step {time_step}")
    
    scale_ax = [1, len(x)/images.shape[0]]

    for i in range(2):
        for j in range(3):
            axs[i,j].set_aspect("equal")
         
            axs[i,j].set_yticks(np.linspace(0, scale_ax[i]*images.shape[0]-1, 8))
            axs[i,j].set_yticklabels(map(int, np.linspace(0, pixels2um*(images.shape[0]-1), 8)))
            axs[i,j].set_xticks(np.linspace(0, scale_ax[i]*images.shape[1]-1, 5))
            axs[i,j].set_xticklabels(map(int, np.linspace(0, pixels2um*(images.shape[1]-1), 8)))
            axs[i,j].set_xlabel("$\mu m$")
            axs[i,j].set_ylabel("$\mu m$")
    
    return magnitude, normalized, Q0, Q1, Q2, Q3, Q4, Q5, fig


def animate_vectorfield(vectors, label, unit, pixels2um, images, fname="animation", \
        framerate=None, extension="mp4", dpi=300, dx=3, scale=1, linscale=1):

    extensions = ["gif", "mp4"]
    msg = f"Invalid extension {extension}. Expected one of {extensions}"
    assert extension in extensions, msg

    magnitude, normalized, Q0, Q1, Q2, Q3, Q4, Q5, fig = \
            plot2Dvalues(vectors, 0, label, unit, dx, dpi, pixels2um, scale, images, linscale)
    
    def update(idx):
        Q0.set_array(images[:,:,idx])
        Q1.set_UVC(vectors[idx, ::dx, ::dx, 1], \
                -vectors[idx, ::dx, ::dx, 0])
        Q2.set_UVC(normalized[idx, ::dx, ::dx, 1], \
                -normalized[idx, ::dx, ::dx, 0])
        Q3.set_data(magnitude[idx, :, :, 0])
        Q4.set_data(vectors[idx, :, :, 0])
        Q5.set_data(vectors[idx, :, :, 1])

        plt.suptitle(f"Time step {idx}")


    # Set up formatting for the movie files
    if extension == "mp4":
        Writer = animation.writers["ffmpeg"]
    else:
        Writer = animation.writers["imagemagick"]
    writer = Writer(fps=framerate)

    N = vectors.shape[0]
    anim = animation.FuncAnimation(fig, update, N)

    fname = os.path.splitext(fname)[0]
    anim.save(f"{fname}.{extension}", writer=writer)


def plot_at_peak(vectors, label, unit, pixels2um, images, fname="vector_fields", \
        extension="png", dpi=600, dx=3, scale=1, linscale=1):

    D = vectors.shape[-1]
    peak = np.argmax(calc_norm_over_time(vectors))

    if D==2:
        plot2Dvalues(vectors, peak, label, unit, dx, dpi, pixels2um, scale, images, linscale)
        filename = fname + "." + extension
        plt.savefig(filename)

def visualize_vectorfield(f_in, layers, framerate_scale=1, save_data=True):
    """

    Visualize fields - "main function"

    """
    layers = layers.split(" ")

    mt_data = mps.MPS(f_in)
    pixels2um = mt_data.info["um_per_pixel"]

    for layer in layers:
        layer_fn = eval(layer)
    
        output_folder = os.path.join(\
                make_dir_layer_structure(f_in, \
                "visualize_vectorfield"), layer)
        make_dir_structure(output_folder)

        data = read_prev_layer(f_in, layer, layer_fn, save_data)

        linscales = {"displacement" : 0.5*np.max(data["all_values"]["displacement"]), \
                      "principal strain" : 0.001*np.max(data["all_values"]["principal strain"]), \
                      "velocity" : 0.25*np.max(data["all_values"]["velocity"])}

        scales = {"displacement" : 6E-4, \
                      "principal strain" : 2E-4, \
                      "velocity" : 1E-5}

        for key in data["all_values"].keys():
            if data["all_values"][key].shape[-1] != 2:
                continue

            unit = data["units"][key]
            print("Plots for " + key + " ...")
            plot_at_peak(data["all_values"][key], key, unit, pixels2um, mt_data.data.frames, \
                    fname=os.path.join(output_folder, "vectorfield_" + key),
                    scale=scales[key], linscale=linscales[key])

            animate_vectorfield(data["all_values"][key], key, unit, pixels2um, \
                    mt_data.data.frames, framerate=framerate_scale*mt_data.framerate, \
                    fname=os.path.join(output_folder, "vectorfield_" + key),\
                    scale=scales[key], linscale=linscales[key])
