"""

Åshild Telle / Simula Research Laboratory / 2019

"""

import numpy as np
import matplotlib.pyplot as plt

from ..utils.bf_mps import BFMPS

from ..utils.data_layer import generate_filename
from ..utils.data_layer import read_prev_layer
from ..mechanical_analysis.mechanical_analysis import (
    analyze_mechanics,
)


def load_input_data(f_in, param_list, overwrite_all):
    """
    
    Reads in / loads information from input files.

    Args:
        f_in - BF / nd2 file
        param_list - parameters given at command line
        overwrite_all - boolean value; recalculate previous layers or not

    Returns:
        mps_data - MPS object
        mc_data - dictionary with results from mechanical analysis

    """

    mps_data = BFMPS(f_in)

    mc_data = read_prev_layer(
        f_in, analyze_mechanics, param_list[:-1], overwrite_all
    )

    return mps_data, mc_data


def make_pretty_label(key, unit):
    """

    Consistent labels with capital letters, no underscores and units in ().

    Args:
        key - which quantity
        unit - corresponding unit

    """

    label = key.capitalize()
    return label.replace("_", " ") + f" ({unit})"


def setup_frame(num_rows, num_cols, sharex, sharey):
    """

    Setup of subplots, gives similar configuration across different
    kinds of plots. Aligns axes.

    Args:
        num_rows - number of rows among the subplots
        num_cols - number of columns

    """

    figsize = (14, 12)
    dpi = 300
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        sharex=sharex,
        sharey=sharey,
        figsize=figsize,
        dpi=dpi,
        squeeze=False,
    )

    axes = axes.flatten()
    fig.align_ylabels(axes)

    return axes, fig


def get_plot_fun(values, corr_plots):
    """

    Visualization/plotting scripts are usually performed over 1D, 2D
    or 4D values (shape (), (2,) or (2, 2)). Given one of these
    shapes, and a list of corresponding functions which performs the
    plotting, this functions performs the mapping between these two.

    Args:
        values - to be plotted, numpy array of dim. T x X x Y X D
        corr_plots - list of 3 different plotting functions

    """

    num_dims = values.shape[3:]

    assert num_dims in (
        (),
        (2,),
        (2, 2),
    ), f"Error: shape of {num_dims} not recognized."

    plot_1d_values, plot_2d_values, plot_4d_values = corr_plots

    if num_dims == ():
        return plot_1d_values

    if num_dims == (2,):
        return plot_2d_values

    return plot_4d_values


def make_quiver_plot(axis, values, coords, color, scale):
    """

    Quiver plots for 2D values.

    """

    assert values.shape[2:] == (
        2,
    ), f"Error: Given value shape ({values.shape[2:]}) do not corresponds to vector values."

    axis.invert_yaxis()

    return axis.quiver(
        coords[1],
        coords[0],
        values[:, :, 1],
        -values[:, :, 0],
        color=color,
        units="xy",
        headwidth=6,
        scale=scale,
    )


def make_heatmap_plot(axis, scalars, vmin, vmax, cmap):
    """

    Gives a heatmap based on magnitude given in scalars.

    Args:
        axis - defines subplot
        scalars - values; X x Y numpy array
        vmin - minimum value possible
        vmax - maximum value possible
        cmap - type of colour map

    """

    return axis.imshow(scalars, vmin=vmin, vmax=vmax, cmap=cmap, alpha=0.5)


def setup_for_key(mps_data, mc_data, key):
    """

    Some possible useful way to structure the input data.

    Args:
        mps_data - output from mps script
        mc_data - output from mpsmechanics/analyze_mechanics script
        key - which value to track for; key for mc_data dicts

    Return:
        metadata - dict with info for labels, axes, etc.
        spatial_data - dict with spatial information (values, images)
        time - numpy array

    """

    images = np.moveaxis(mps_data.frames, 2, 0)
    time = mc_data["time"]

    values = mc_data["all_values"][key]

    metadata = {
        "label": make_pretty_label(key, mc_data["unit"][key]),
        "pixels2um": mps_data.info["um_per_pixel"],
        "blocksize": images.shape[1] // values.shape[1],
    }

    spatial_data = {"images": images, "derived_quantity": values}

    return metadata, spatial_data, time


def generate_filenames_pngmp4(f_in, subfolder, prefix, param_list):
    
    param_list_copy = [_p.copy() for _p in param_list]
    
    scaling_factor = str(param_list_copy[-1].pop("scaling_factor"))
    scaling_factor = scaling_factor.replace(".", "p")
     
    time_step = param_list_copy[-1].pop("time_step") 

    fname = generate_filename(
        f_in,
        prefix,
        param_list_copy,
        "",                   # mp3 or png
        subfolder=subfolder,
    )

    if time_step is not None:
        fname_png = f"{fname}__time_step_{time_step}.png"
    else:
        fname_png = f"{fname}_at_peak.png"

    fname_mp4 = f"{fname}__animation_scaling_factor_{scaling_factor}.mp4"

    return fname_png, fname_mp4
