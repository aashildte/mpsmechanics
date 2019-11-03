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


def plt_magnitude(axis, i, scalars, vmin, vmax, cmap, scale, alpha=1):
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

    return axis.imshow(scalars[i, :, :], vmin=vmin, vmax=vmax, cmap=cmap, norm=norm, alpha=alpha)


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


def plot_1Dvalues(values, scale, time, time_step, label, dpi, pixels2um, images): 
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
    
    plt.suptitle("Time: {} ms".format(int(time[time_step])))

    def update(index):
        subplots[0].set_array(images[:, :, index])
        subplots[1].set_data(values[index, :, :, 0])
        plt.suptitle("Time: {} ms".format(int(time[index])))

    return fig, update


def plot_2Dvalues(values, scale_magnitude, time, time_step, label, dpi, pixels2um, images, num_arrows=3):
    
    scale_arrows = _find_arrow_scaling(values, num_arrows)

    magnitude = calc_magnitude(values)
    normalized = normalize_values(values)

    x, y, axes, fig = setup_frame(values, dpi, images, 2, 3)

    block_size = len(x) // values.shape[1]
    scale_n = max(np.divide(values.shape[1:3], block_size))

    scale_xy = np.max(np.abs(values))
    
    subplots = []

    subplots.append(axes[0].imshow(images[:, :, time_step], cmap=cm.gray))
    subplots.append(plt_quiver(axes[1], time_step, values, x, y, num_arrows, 'red', scale_arrows))
    subplots.append(plt_quiver(axes[2], time_step, normalized, x, y, num_arrows, 'black', None))
    subplots.append(plt_magnitude(axes[3], time_step, magnitude[:, :, :, 0], 0, np.max(magnitude), "viridis", scale_magnitude))
    cb = fig.colorbar(subplots[3], ax=axes[3])
    cb.set_label(label)

    subplots.append(plt_magnitude(axes[4], time_step, values[:, :, :, 0], -scale_xy, scale_xy, "bwr", scale_magnitude))
    cb = fig.colorbar(subplots[4], ax=axes[4])
    cb.set_label(label)

    subplots.append(plt_magnitude(axes[5], time_step, values[:, :, :, 1], -scale_xy, scale_xy, "bwr", scale_magnitude))
    cb = fig.colorbar(subplots[5], ax=axes[5])
    cb.set_label(label)

    titles = ["Original images", "Vector field", "Direction", "Magnitude", "Longitudinal (x)", "Transversal (y)"]
    for (title, axis) in zip(titles, axes):
        axis.set_title(title)

    plt.suptitle("Time: {} ms".format(int(time[time_step])))

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
        
        plt.suptitle("Time: {} ms".format(int(time[index])))


    return fig, update


def plot_vectorfield(values, time, time_step, label, num_arrows, dpi, pixels2um, images):
    x, y, axes, fig = setup_frame(values, dpi, images, 1, 1)

    block_size = len(x) // values.shape[1]
    scale_n = max(np.divide(values.shape[1:3], block_size))

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


def plot_1Dvalues_overlap(values, scale, time, time_step, label, dpi, pixels2um, images):
    x, y, axes, fig = setup_frame(values, dpi, images, 1, 1)
    subplots = []
    
    subplots.append(axes[0].imshow(images[:, :, time_step], cmap=cm.gray))
    subplots.append(plt_magnitude(axes[0], time_step, values[:, :, :, 0], \
                       0, np.max(values), "viridis", scale, alpha=0.5))
    cb = fig.colorbar(subplots[1], ax=axes[0])
    cb.set_label(label)

    axes[0].set_title("Original images")
    axes[0].set_title("Magnitude")
    plt.suptitle("Time: {} ms".format(time[time_step]))
    
    _set_ax_units(axes[0], images.shape[:2], pixels2um)

    def update(index):
        subplots[0].set_array(images[:, :, index])
        subplots[1].set_data(values[index, :, :, 0])
        
        plt.suptitle("Time: {} ms".format(int(time[index])))

    return fig, update



def plot_4Dvalues(values, scale, time, time_step, label, dpi, pixels2um, images):
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

    plt.suptitle("Time: {} ms".format(int(time[time_step])))

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
    
        plt.suptitle("Time: {} ms".format(int(time[index])))

    return fig, update


def animate_decomposition(values, time, scale_magnitude, label, pixels2um, images, fname, \
        framerate=None, extension="mp4", dpi=300, num_arrows=3):

    plot_fn = get_plot_fn(values)

    extensions = ["gif", "mp4"]
    msg = "Invalid extension {}. Expected one of {}".format(extension, extensions)
    assert extension in extensions, msg

    fig, update = plot_fn(values, scale_magnitude, time, 0, label, dpi, \
                        pixels2um, images)

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


def plot_vectorfield_at_peak(values, time, label, pixels2um, images, fname, \
                        extension="mp4", dpi=300, num_arrows=3):
    
    num_dims = values.shape[3:] 
    assert num_dims == (2,), "Error: T x X x Y x 2 numpy array expected as first argument."
    
    peak = np.argmax(calc_norm_over_time(values))
         
    plot_vectorfield(values, time, peak, label, num_arrows, dpi, pixels2um, images)

    filename = fname + ".png"
    plt.savefig(filename)
    plt.close('all')
    

def animate_vectorfield(values, time, label, pixels2um, images, fname, \
                        framerate=None, extension="mp4", dpi=300, num_arrows=3):
    
    num_dims = values.shape[3:] 
    assert num_dims == (2,), "Error: T x X x Y x 2 numpy array expected as first argument."
    
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
    

def get_plot_fn(values):
    num_dims = values.shape[3:]
    
    if num_dims == (1,):
        return plot_1Dvalues
    
    if num_dims == (2,):
        return plot_2Dvalues
    
    if num_dims == (2, 2):
        return plot_4Dvalues
    
    print("Error: shape of {} not recognized.".format(num_dims))


def plot_decomposition_at_peak(values, time, scale_magnitude, label, pixels2um, images, fname, \
        extension="png", dpi=600, num_arrows=3):

    peak = np.argmax(calc_norm_over_time(values))
    plot_fn = get_plot_fn(values)
    plot_fn(values, scale_magnitude, time, peak, label, dpi, \
            pixels2um, images)

    filename = fname + "." + extension
    plt.savefig(filename)
    plt.close('all')


def _make_vectorfield_plots(values, time, key, label, output_folder, pixels2um, \
        images, framerate, animate, sigma):
    num_dims = values.shape[3:] 
    if num_dims != (2,):
        return
     
    fname = os.path.join(output_folder, f"vectorfield_{key}_{sigma}_{sigma}_{sigma}")
    plot_vectorfield_at_peak(values, time, label, pixels2um, \
                    images, fname)

    if animate:
        print("Making a movie ..")
        animate_vectorfield(values, time, label, pixels2um, \
                    images, fname, framerate=framerate)
    
    
def _make_decomposition_plots(values, time, key, label, output_folder, pixels2um, \
        images, framerate, animate, sigma):

    for scale_magnitude in ["logscale", "linear"]:
        fname = os.path.join(output_folder, f"spatial_{scale_magnitude}_{key}_{sigma}_{sigma}_{sigma}")
        plot_decomposition_at_peak(values, time, scale_magnitude, label, pixels2um, images, \
                        fname)
        
        if animate:
            print("Making a movie ..")
            
            animate_decomposition(values, time, scale_magnitude, label, pixels2um, \
                        images, fname, framerate=framerate) 


def visualize_mechanics_spatial(f_in, framerate_scale, animate=False, overwrite=False, save_data=True):
    """

    Visualize fields - "main function"

    """
    
    mt_data = mps.MPS(f_in)
    pixels2um = mt_data.info["um_per_pixel"]
    images = mt_data.data.frames
    framerate = mt_data.framerate

    output_folder = make_dir_layer_structure(f_in, \
            os.path.join("mpsmechanics", "visualize_vectorfield"))
    make_dir_structure(output_folder)
    

    for size in [0, 1, 2, 3, 4, 5, 10, 15]:
        sigma = 0.1*size
        mc_data = read_prev_layer(f_in, f"analyze_mechanics_{sigma}_{sigma}_{sigma}", analyze_mechanics, save_data)
        time = mc_data["time"]

        for key in mc_data["all_values"].keys():
            print("Plots for " + key + " ...")
            label = key.capitalize() + "({})".format(mc_data["units"][key])
            label.replace("_", " ")

            values = mc_data["all_values"][key]

            _make_vectorfield_plots(values, time, key, label, output_folder, pixels2um, \
                    images, framerate, animate, sigma)

            _make_decomposition_plots(values, time, key, label, output_folder, pixels2um, \
                    images, framerate_scale*framerate, animate, sigma)

    print("Visualization done, finishing ..")

