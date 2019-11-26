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
    


def _make_vectorfield_plots(values, time, key, label, output_folder, pixels2um, \
        images, framerate, animate, fname):
    num_dims = values.shape[3:] 
    if num_dims != (2,):
        return
     
    plot_vectorfield_at_peak(values, time, label, pixels2um, \
                    images, fname)

    if animate:
        print("Making a movie ..")
        animate_vectorfield(values, time, label, pixels2um, \
                    images, fname, framerate=framerate)
    
    

def visualize_vectorfield(f_in, overwrite, param_list):
    """

    Visualize fields - "main function"

    """
    
    print("Parameters visualization of vectorfield:")
    for key in param_list[2].keys():
        print(" * {}: {}".format(key, param_list[2][key]))
    
    mc_data = read_prev_layer(
        f_in,
        analyze_mechanics,
        param_list[:-1],
        overwrite
    )
    
    animate = param_list[2]["animate"]
    scaling_factor = param_list[2]["scaling_factor"]
    
    mt_data = mps.MPS(f_in)
    pixels2um = mt_data.info["um_per_pixel"]
    images = mt_data.data.frames
    framerate = mt_data.framerate

    output_folder = make_dir_layer_structure(f_in, \
            os.path.join("mpsmechanics", "visualize_vectorfield"))
    os.makedirs(output_folder, exist_ok=True)
    
    time = mc_data["time"]

    for key in mc_data["all_values"].keys():
        print("Plots for " + key + " ...")
        label = key.capitalize() + "({})".format(mc_data["units"][key])
        label.replace("_", " ")

        fname = generate_filename(f_in, \
                                  os.path.join("visualize_vectorfield", f"spatial_{key}"), \
                                  param_list[:2])

        values = mc_data["all_values"][key]

        _make_vectorfield_plots(values, time, key, label, output_folder, pixels2um, \
                images, scaling_factor*framerate, animate, fname)

    print("Visualization done, finishing ..")

