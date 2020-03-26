"""

Åshild Telle / Simula Research Laboratory / 2019

"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..utils.data_layer import generate_filename
from ..dothemaths.operations import calc_magnitude, \
        normalize_values, calc_norm_over_time
from .animation_funs import make_animation, \
        get_animation_configuration
from .setup_plots import setup_frame, get_plot_fun, load_input_data, \
        make_quiver_plot, make_heatmap_plot, setup_for_key

def _set_ax_units(axis, scale, shift):

    axis.set_aspect("equal")
    num_yticks = 8
    num_xticks = 4

    y_range = axis.get_ylim()
    x_range = axis.get_xlim()

    y_to = y_range[0] #- y_range[1]
    x_to = x_range[1] #- x_range[0]

    yticks = np.linspace(0, y_to, num_yticks)
    ylabels = scale*yticks + shift

    xticks = np.linspace(0, x_to, num_xticks)
    xlabels = scale*xticks + shift

    xlabels = xlabels.astype(int)
    ylabels = ylabels.astype(int)

    axis.set_yticks(yticks)
    axis.set_xticks(xticks)
    axis.set_yticklabels(ylabels)
    axis.set_xticklabels(xlabels)

    axis.set_xlabel(r"$\mu m$")
    axis.set_ylabel(r"$\mu m$")


def _find_arrow_scaling(values):
    return 0.5*np.mean(np.abs(values))


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


def _init_subplots_1d(all_components, time_step):
    axes, fig = setup_frame(1, 2, False, False)
    subplots = []

    images, magnitude = all_components
    subplots.append(axes[0].imshow(images[time_step], cmap="gray"))
    subplots.append(make_heatmap_plot(axes[1], magnitude[time_step], \
                       0, np.max(magnitude), "viridis"))

    return axes, fig, subplots


def _make_1d_plot_pretty(fig, axes, subplots, subtitles, metadata):
    label = metadata["label"]
    pixels2um = metadata["pixels2um"]
    blocksize = metadata["blocksize"]

    fig.colorbar(subplots[1], ax=axes[1]).set_label(label)

    for (axis, subtitle) in zip(axes, subtitles):
        axis.set_title(subtitle)

    _set_ax_units(axes[0], pixels2um, 0)
    _set_ax_units(axes[1], blocksize*pixels2um, blocksize//2)


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
    axes, fig, subplots = _init_subplots_1d(all_components, time_step)
    _make_1d_plot_pretty(fig, axes, subplots, subtitles, metadata)
    plt.suptitle("Time: {} ms".format(int(time[time_step])))

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


def _find_xy_coords(images, values):
    x_len, y_len = images.shape[1:3]
    x_coords = np.linspace(0, x_len, values.shape[1])
    y_coords = np.linspace(0, y_len, values.shape[2])

    return np.asarray([x_coords, y_coords])


def _init_subplots_2d(all_components, time_step):
    axes, fig = setup_frame(2, 3, False, False)

    images, values, normalized, magnitude, x_values, y_values = \
            all_components

    vmin, vmax = _get_value_range([x_values, y_values])

    coords = _find_xy_coords(images, values)

    vmax_mag = np.max(magnitude)
    subplots = [axes[0].imshow(images[time_step], cmap="gray"),
                make_quiver_plot(axes[1], values[time_step], coords, \
                           'red', _find_arrow_scaling(values)),
                make_quiver_plot(axes[2], normalized[time_step], coords, \
                           'black', 0.05),
                make_heatmap_plot(axes[3], magnitude[time_step], 0, \
                              vmax_mag, "viridis"),
                make_heatmap_plot(axes[4], x_values[time_step], \
                              vmin, vmax, "bwr"),
                make_heatmap_plot(axes[5], y_values[time_step], \
                              vmin, vmax, "bwr")]

    return axes, fig, subplots


def _make_2d_plot_pretty(fig, axes, subplots, subtitles, metadata):
    label = metadata["label"]
    pixels2um = metadata["pixels2um"]
    blocksize = metadata["blocksize"]

    for axis in axes[:3]:
        _align_subplot(axis)
        axis.set_aspect('equal')

    for i in range(3, 6):
        fig.colorbar(subplots[i], ax=axes[i]).set_label(label)

    for (subtitle, axis) in zip(subtitles, axes):
        axis.set_title(subtitle)

    _set_ax_units(axes[0], pixels2um, 0)

    for axis in axes[1:]:
        _set_ax_units(axis, blocksize*pixels2um, blocksize//2)

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
    axes, fig, subplots = _init_subplots_2d(all_components, time_step)
    _make_2d_plot_pretty(fig, axes, subplots, subtitles, metadata)

    plt.suptitle("Time: {} ms".format(int(time[time_step])))

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


def _init_subplots_2x2d(all_components, time_step, shift):
    axes, fig = setup_frame(2, 3, False, False)

    images, ux_values, uy_values, \
            sg_values, vx_values, vy_values = all_components

    val_range = _get_value_range([ux_values, uy_values, \
                                   vx_values, vy_values])

    sg_range = np.min(sg_values), np.max(sg_values)

    subplots = [axes[0].imshow(images[time_step], cmap="gray"),
                make_heatmap_plot(axes[1], ux_values[time_step], \
                    val_range[0] + int(shift), val_range[1] + int(shift), "bwr"),
                make_heatmap_plot(axes[2], uy_values[time_step], \
                    val_range[0], val_range[1], "bwr"),
                make_heatmap_plot(axes[3], sg_values[time_step], \
                        sg_range[0], sg_range[1], "viridis"),
                make_heatmap_plot(axes[4], vx_values[time_step], \
                    val_range[0], val_range[1], "bwr"),
                make_heatmap_plot(axes[5], vy_values[time_step], \
                    val_range[0] + int(shift), val_range[1] + int(shift), "bwr")]

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

    _set_ax_units(axes[0], pixels2um, 0)
    for axis in axes[1:]:
        _set_ax_units(axis, blocksize*pixels2um, blocksize//2)


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
    axes, fig, subplots = \
            _init_subplots_2x2d(all_components, \
                                time_step, \
                                metadata["shift_diagonal"])
    _make_2x2d_plot_pretty(fig, \
                           axes, \
                           subplots, \
                           subtitles, \
                           metadata)
    plt.suptitle("Time: {} ms".format(int(time[time_step])))

    def _update(index):
        for (subplot, component) in zip(subplots, all_components):
            subplot.set_data(component[index])

        plt.suptitle("Time: {} ms".format(int(time[index])))

    return fig, _update


def _plot_at_peak(spatial_data, time, metadata, fname):

    values = spatial_data["derived_quantity"]

    peak = np.argmax(calc_norm_over_time(values))
    plot_fn = get_plot_fun(values, \
            [plot_1d_values, plot_2d_values, plot_2x2d_values])
    plot_fn(spatial_data, time, peak, metadata)

    plt.savefig(fname)
    plt.close('all')


def _make_animation(spatial_data, time, metadata, fname, animation_config):
    plot_fn = get_plot_fun(spatial_data["derived_quantity"], \
            [plot_1d_values, plot_2d_values, plot_2x2d_values])
    fig, _update = plot_fn(spatial_data, time, 0, metadata)

    make_animation(fig, _update, fname, **animation_config)


def _make_filenames(f_in, metric):
    fname = generate_filename(f_in, \
                              f"spatial_decomposition_{metric}", \
                              [],
                              "",        # mp3 or png
                              subfolder="visualize_mechanics")
    fname_png = fname + ".png"
    fname_mp4 = fname + ".mp4"

    return fname_png, fname_mp4


def visualize_mechanics(f_in, overwrite, overwrite_all, param_list):
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

    mps_data, mc_data = load_input_data(f_in, param_list, overwrite_all)
    animation_config = get_animation_configuration(param_list[-1], mps_data)
    animate = animation_config.pop("animate")


    metrics = param_list[-1]["metrics"].split(" ")

    for metric in metrics:
        assert metric in mc_data["all_values"].keys(), \
                f"Error: Metric expected to be in {mc_data['all_values'].keys()}"        

        print("Making plot for " + metric + " ...")

        fname_png, fname_mp4 = _make_filenames(f_in, metric)
        metadata, spatial_data, time = setup_for_key(mps_data, mc_data, metric)

        metadata["shift_diagonal"] = (metric == "deformation_tensor")

        if overwrite or (not os.path.isfile(fname_png)):
            _plot_at_peak(spatial_data, time, metadata, fname_png)
            print("Plot at peak done; " + \
                  f"image saved to {fname_png}")
        else:
            print(f"Image {fname_png} already exists")

        if animate:
            if (overwrite or (not os.path.isfile(fname_mp4))):
                _make_animation(spatial_data, time, metadata, fname_mp4, \
                                animation_config)
                print("Animation movie produced; " + \
                      f"movie saved to {fname_mp4}")
            else:
                print(f"Movie {fname_mp4} already exists")

        print("Visualization done, finishing ...")
