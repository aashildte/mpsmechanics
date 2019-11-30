"""

Åshild Telle / Simula Research Laboratory / 2019

"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import mps

from ..utils.data_layer import read_prev_layer, generate_filename
from ..dothemaths.operations import calc_magnitude, normalize_values, calc_norm_over_time
from ..mechanical_analysis.mechanical_analysis import analyze_mechanics
from .animation_funs import make_animation
from .setup_plots import setup_frame, get_plot_fun, make_pretty_label


def make_quiver_plot(axis, values, quiver_step, color, scale):
    """

    Quiver plots for 2D values.

    """

    assert values.shape[2:] == (2,), \
            f"Error: Given value shape ({values.shape[2:]}) do not corresponds to vector values."

    coords = [np.linspace(0, quiver_step*values.shape[1], values.shape[1]),
              np.linspace(0, quiver_step*values.shape[2], values.shape[2])]

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
    return axis.imshow(scalars, vmin=vmin, vmax=vmax, \
            cmap=cmap)


def _set_ax_units(axis, scale):

    axis.set_aspect("equal")
    num_yticks = 8
    num_xticks = 4

    y_to, _ = axis.get_ylim()
    _, x_to = axis.get_xlim()

    yticks = np.linspace(0, y_to, num_yticks)
    ylabels = scale*yticks

    xticks = np.linspace(0, x_to, num_xticks)
    xlabels = scale*xticks

    xlabels = xlabels.astype(int)
    ylabels = ylabels.astype(int)

    axis.set_yticks(yticks)
    axis.set_xticks(xticks)
    axis.set_yticklabels(ylabels)
    axis.set_xticklabels(xlabels)

    axis.set_xlabel(r"$\mu m$")
    axis.set_ylabel(r"$\mu m$")


def _find_arrow_scaling(values):
    return 1.5*np.mean(np.abs(values))


def _get_value_range(values):
    lim = np.max(np.abs(np.asarray(values)))
    return -lim, lim


def _align_subplot(axis):
    divider = make_axes_locatable(axis)
    cax = divider.append_axes("left", size="0%", pad=0.495)
    cax.remove()


def _get_1d_values(images, values):
    all_components = [images, values]
    subtitles = ["Original image", "Magnitude"]

    return all_components, subtitles


def _init_subplots_1d(all_components, time, time_step):
    axes, fig = setup_frame(1, 2, False, False)
    subplots = []

    images, magnitude = all_components
    subplots.append(axes[0].imshow(images[time_step], cmap="gray"))
    subplots.append(make_heatmap_plot(axes[1], magnitude[time_step], \
                       0, np.max(magnitude), "viridis"))

    plt.suptitle("Time: {} ms".format(int(time[time_step])))

    return axes, fig, subplots


def _make_1d_plot_pretty(fig, axes, subplots, subtitles, metadata):
    label = metadata["label"]
    pixels2um = metadata["pixels2um"]
    blocksize = metadata["blocksize"]

    fig.colorbar(subplots[1], ax=axes[1]).set_label(label)

    for (axis, subtitle) in zip(axes, subtitles):
        axis.set_subtitle(subtitle)

    _set_ax_units(axes[0], pixels2um)
    _set_ax_units(axes[1], blocksize*pixels2um)


def plot_1d_values(spatial_data, time, time_step, metadata):
    """

    Plots original image and magnitude.

    Args:
        spatial_data - dictionary
            - images : original BF images
            - derived_quantity : T x X x Y numpy array
        time - corresponding time units
        time_step - integer value; time step of interest, typically peak or 0
        metadata - dictionary with information about labels, units

    """

    images = spatial_data["images"]
    values = spatial_data["derived_quantity"]

    all_components, subtitles = _get_1d_values(images, values)
    axes, fig, subplots = _init_subplots_1d(all_components, time, time_step)
    _make_1d_plot_pretty(fig, axes, subplots, subtitles, metadata)

    def _update(index):
        subplots[0].set_array(images[index])
        subplots[1].set_data(values[index])
        plt.suptitle("Time: {} ms".format(int(time[index])))

    return fig, _update


def _get_2d_values(images, values, quiver_step):
    downsampled = values[:, ::quiver_step, ::quiver_step]
    normalized = normalize_values(values)[:, ::quiver_step, ::quiver_step]
    magnitude = calc_magnitude(values)
    x_values = values[:, :, :, 0]
    y_values = values[:, :, :, 1]

    all_components = [images, downsampled, normalized, \
                        magnitude, x_values, y_values]
    titles = ["Original image", "Vector field", "Direction", \
                "Magnitude", "Longitudinal (x)", "Transversal (y)"]

    return all_components, titles


def _init_subplots_2d(all_components, time, time_step, quiver_step):
    axes, fig = setup_frame(2, 3, False, False)

    images, values, normalized, magnitude, x_values, y_values = \
            all_components

    vmin, vmax = _get_value_range([x_values, y_values])

    subplots = [axes[0].imshow(images[time_step], cmap="gray"),
                make_quiver_plot(axes[1], values[time_step], quiver_step, \
                           'red', _find_arrow_scaling(values)),
                make_quiver_plot(axes[2], normalized[time_step], quiver_step, \
                           'black', 1.5/quiver_step),
                make_heatmap_plot(axes[3], magnitude[time_step], 0, \
                              np.max(magnitude), "viridis"),
                make_heatmap_plot(axes[4], x_values[time_step], \
                              vmin, vmax, "bwr"),
                make_heatmap_plot(axes[5], y_values[time_step], \
                              vmin, vmax, "bwr")]

    plt.suptitle("Time: {} ms".format(int(time[time_step])))

    return axes, fig, subplots


def _make_2d_plot_pretty(fig, axes, subplots, subtitles, metadata):
    label = metadata["label"]
    pixels2um = metadata["pixels2um"]
    blocksize = metadata["blocksize"]

    for axis in axes[:3]:
        _align_subplot(axis)

    for i in range(3, 6):
        fig.colorbar(subplots[i], ax=axes[i]).set_label(label)

    for (subtitle, axis) in zip(subtitles, axes):
        axis.set_title(subtitle)

    _set_ax_units(axes[0], pixels2um)

    for axis in axes[3:]:
        _set_ax_units(axis, blocksize*pixels2um)

def plot_2d_values(spatial_data, time, time_step, metadata):
    """

    Plots original image, vector field, normalized vector field,
    magnitude, magnitude x component and magnitude y component.

    Args:
        spatial_data - dictionary
            - images : original BF images
            - derived_quantity : T x X x Y x 2 numpy array
        time - corresponding time units
        time_step - integer value; time step of interest, typically peak or 0
        metadata - dictionary with information about labels, units

    """

    images = spatial_data["images"]
    values = spatial_data["derived_quantity"]

    quiver_step = 3
    all_components, subtitles = _get_2d_values(images, values, quiver_step)
    axes, fig, subplots = _init_subplots_2d(all_components, time, time_step, quiver_step)
    _make_2d_plot_pretty(fig, axes, subplots, subtitles, metadata)

    def _update(index):
        for i in (0, 3, 4, 5):
            subplots[i].set_data(all_components[i][index])

        for i in (1, 2):
            subplots[i].set_UVC(all_components[i][index, :, :, 1],
                                -all_components[i][index, :, :, 0])

        plt.suptitle("Time: {} ms".format(int(time[index])))

    return fig, _update


def _get_2x2d_values(images, values):
    ux_values = values[:, :, :, 0, 0]
    uy_values = values[:, :, :, 0, 1]
    sg_values = np.linalg.norm(values, axis=(3, 4))
    vx_values = values[:, :, :, 1, 0]
    vy_values = values[:, :, :, 1, 1]

    all_components = [images, ux_values, uy_values, \
            sg_values, vx_values, vy_values]
    subtitles = ["Original image", r"$u_x$", r"$u_y$", \
            "Largest singular value", r"$v_x$", r"$v_y$"]

    return all_components, subtitles


def _init_subplots_2x2d(all_components, time, time_step):
    images, ux_values, uy_values, \
            sg_values, vx_values, vy_values = all_components
    axes, fig = setup_frame(2, 3, False, False)

    vmin, vmax = _get_value_range([ux_values, uy_values, \
                                   vx_values, vy_values])

    subplots = [axes[0].imshow(images[time_step], cmap="gray"),
                make_heatmap_plot(axes[1], ux_values[time_step], \
                    vmin, vmax, "bwr"),
                make_heatmap_plot(axes[2], uy_values[time_step], \
                    vmin, vmax, "bwr"),
                make_heatmap_plot(axes[3], sg_values[time_step], \
                        np.min(sg_values), np.max(sg_values), "viridis"),
                make_heatmap_plot(axes[4], vx_values[time_step], \
                    vmin, vmax, "bwr"),
                make_heatmap_plot(axes[5], vy_values[time_step], \
                    vmin, vmax, "bwr")]

    plt.suptitle("Time: {} ms".format(int(time[time_step])))

    return axes, fig, subplots


def _make_2x2d_plot_pretty(fig, axes, subplots, subtitles, metadata):
    label = metadata["label"]
    pixels2um = metadata["pixels2um"]
    blocksize = metadata["blocksize"]

    for i in range(1, 6):
        fig.colorbar(subplots[i], ax=axes[i]).set_label(label)

    for (axis, subtitle) in zip(axes, subtitles):
        axis.set_title(subtitle)

    _align_subplot(axes[0])

    _set_ax_units(axes[0], pixels2um)
    for axis in axes[1:]:
        _set_ax_units(axis, blocksize*pixels2um)


def plot_2x2d_values(spatial_data, time, time_step, metadata):
    """

    Plots original image, eigenvalues (max.) and four components of
    a tensor value.

    Args:
        spatial_data - dictionary
            - images : original BF images
            - derived_quantity : T x X x Y x 2 x 2 numpy array
        time - corresponding time units
        time_step - integer value; time step of interest, typically peak or 0
        metadata - dictionary with information about labels, units

    """

    images = spatial_data["images"]
    values = spatial_data["derived_quantity"]

    all_components, subtitles = _get_2x2d_values(images, values)
    axes, fig, subplots = _init_subplots_2x2d(all_components, time, time_step)
    _make_2x2d_plot_pretty(fig, axes, subplots, subtitles, metadata)

    def _update(index):
        for (subplot, component) in zip(subplots, all_components):
            subplot.set_data(component[index])

        plt.suptitle("Time: {} ms".format(int(time[index])))

    return fig, _update


def _plot_at_peak(spatial_data, time, metadata, fname):

    extension = "png"
    values = spatial_data["derived_quantity"]

    peak = np.argmax(calc_norm_over_time(values))
    plot_fn = get_plot_fun(values, \
            [plot_1d_values, plot_2d_values, plot_2x2d_values])
    plot_fn(spatial_data, time, peak, metadata)

    filename = fname + "." + extension
    plt.savefig(filename)
    plt.close('all')


def _make_animation(spatial_data, time, metadata, fname, framerate):
    extension = "mp4"
    plot_fn = get_plot_fun(spatial_data["derived_quantity"], \
            [plot_1d_values, plot_2d_values, plot_2x2d_values])
    fig, _update = plot_fn(spatial_data, time, 0, metadata)

    num_frames = len(time)
    make_animation(fig, _update, num_frames, framerate, fname, extension)


def _plot_for_each_key(input_data, param_list, overwrite, animate, scaling_factor):

    f_in, mps_data, mc_data = input_data
    images = np.moveaxis(mps_data.frames, 2, 0)
    time = mc_data["time"]

    for key in mc_data["all_values"].keys():
        print("Plots for " + key + " ...")

        fname = generate_filename(f_in, \
                                  f"decomposition_{key}", \
                                  param_list[:2],
                                  "",        # mp3 or png
                                  subfolder="visualize_mechanics")

        values = mc_data["all_values"][key]

        metadata = {"label" : make_pretty_label(key, mc_data["units"][key]),
                    "pixels2um" : mps_data.info["um_per_pixel"],
                    "blocksize" : images.shape[1] // values.shape[1]}

        spatial_data = {"images" : images,
                        "derived_quantity" : values}

        if not overwrite and not os.path.isfile(fname + ".png"):
            _plot_at_peak(spatial_data, time, metadata, fname)

        if animate and not overwrite and not os.path.isfile(fname + ".mp4"):
            print("Making a movie ..")
            _make_animation(spatial_data, time, metadata, fname, \
                    scaling_factor*mps_data.framerate)


def visualize_mechanics(f_in, overwrite, overwrite_all, param_list):
    """

    "main function"

    Args:
        f_in - BF / nd2 file
        overwrite - recalculate previous data or not
        param_list - list of lists; 3 sublists. First 2 are passed to
            previous layers if data needs to be recalculated; last gives
            parameters for this script.

    """

    print("Parameters visualize distributions:")
    for key in param_list[2].keys():
        print(" * {}: {}".format(key, param_list[2][key]))

    mps_data = mps.MPS(f_in)

    mc_data = read_prev_layer(
        f_in,
        analyze_mechanics,
        param_list[:-1],
        overwrite_all
    )

    input_data = [f_in, mps_data, mc_data]
    _plot_for_each_key(input_data, param_list, overwrite, **param_list[-1])

    print("Visualization done, finishing ...")
