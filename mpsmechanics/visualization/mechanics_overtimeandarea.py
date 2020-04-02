"""

Åshild Telle / Simula Research Laboratory / 2019

"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec


from ..utils.data_layer import generate_filename
from ..dothemaths.operations import (
    calc_magnitude,
    normalize_values,
    calc_norm_over_time,
)
from .animation_funs import (
    make_animation,
    get_animation_configuration,
)
from .setup_plots import (
    setup_frame,
    get_plot_fun,
    load_input_data,
    make_quiver_plot,
    make_heatmap_plot,
    setup_for_key,
)


def setup_frame_gridspec(
    subplots_x, subplots_y, subdivision_x, subdivision_y
):

    fig = plt.figure(figsize=(7 * subplots_y, 18 * subplots_x))
    outer = gridspec.GridSpec(
        subplots_x, subplots_y, wspace=0.2, hspace=0.2
    )

    # one for the image, subdivision_x x subdivision_y for the rest

    inner = gridspec.GridSpecFromSubplotSpec(
        1, 1, subplot_spec=outer[0]
    )
    im_ax = plt.Subplot(fig, inner[0])

    fig.add_subplot(im_ax)

    all_axes = [im_ax]
    shareaxes = None

    for i in range(1, subplots_x * subplots_y):
        inner = gridspec.GridSpecFromSubplotSpec(
            subdivision_x,
            subdivision_y,
            subplot_spec=outer[i],
            hspace=0.05,
            wspace=0.05,
        )

        subplot_axes = []

        for _x in range(subdivision_x):
            subplot_axes_x = []
            for _y in range(subdivision_y):
                if shareaxes is None:
                    ax = plt.Subplot(fig, inner[_x, _y])
                    shareaxes = ax
                else:
                    ax = plt.Subplot(
                        fig,
                        inner[_x, _y],
                        sharex=shareaxes,
                        sharey=shareaxes,
                    )

                if _x < (subdivision_x - 1):
                    ax.get_xaxis().set_visible(False)
                if _y > 0:
                    ax.get_yaxis().set_visible(False)
                fig.add_subplot(ax)
                subplot_axes_x.append(ax)

            subplot_axes.append(subplot_axes_x)
        all_axes.append(subplot_axes)

    return all_axes


def _get_value_range(values):
    lim = np.max(np.abs(np.asarray(values)))
    return -lim, lim


def plot_image_subdivision(
    axis, image, values, subdivisions_xdir, subdivisions_ydir
):
    axis.imshow(image, cmap="gray")

    xshape, yshape = values.shape[1:3]
    xlen, ylen = (
        xshape // subdivisions_xdir,
        yshape // subdivisions_ydir,
    )
    blocksize = image.shape[0] // xshape

    for m in range(0, subdivisions_xdir + 1):
        axis.plot(
            [0, image.shape[1] - 1],
            [m * xlen * blocksize, m * xlen * blocksize],
            "w",
            linewidth=2,
        )

    for n in range(0, subdivisions_ydir + 1):
        axis.plot(
            [n * ylen * blocksize, n * ylen * blocksize],
            [0, image.shape[0] - 1],
            "w",
            linewidth=2,
        )


def plot_over_time(
    axes, values, time, subdivisions_xdir, subdivisions_ydir
):

    assert (
        len(values.shape) == 3
    ), f"Error: Expected shape T x X x Y x 1; received {values.shape}"

    xshape, yshape = values.shape[1:3]

    xlen, ylen = (
        xshape // subdivisions_xdir,
        yshape // subdivisions_ydir,
    )
    area = xlen * ylen

    scale = 0

    for _x in range(subdivisions_xdir):
        for _y in range(subdivisions_ydir):
            xfrom = _x * xlen
            yfrom = _y * ylen
            values_xy = (
                1
                / area
                * np.sum(
                    values[
                        :, xfrom : xfrom + xlen, yfrom : yfrom + ylen
                    ],
                    axis=(1, 2),
                )
            )
            axes[_x][_y].plot(time, values_xy)


def plot_1d_values(
    spatial_data, time, subdivisions_xdir, subdivisions_ydir, label
):
    """

    Plots original image and magnitude.

    """

    image = spatial_data["images"][0]
    values = spatial_data["derived_quantity"]

    axes = setup_frame_gridspec(
        1, 2, subdivisions_xdir, subdivisions_ydir
    )

    plot_image_subdivision(
        axes[0], image, values, subdivisions_xdir, subdivisions_ydir
    )
    plot_over_time(
        axes[1], values, time, subdivisions_xdir, subdivisions_ydir
    )

    axes[1][subdivisions_xdir // 2][0].set_ylabel(label)
    axes[1][-1][subdivisions_ydir // 2].set_xlabel("Time (ms)")

    plt.suptitle(label)


def _get_2d_values(values):
    magnitude = calc_magnitude(values)
    x_values = values[:, :, :, 0]
    y_values = values[:, :, :, 1]

    subtitles = ["magnitude", "longitudinal (x)", "transversal (y)"]

    return magnitude, x_values, y_values, subtitles


def plot_2d_values(
    spatial_data, time, subdivisions_xdir, subdivisions_ydir, label
):
    """

    Plots original image, vector field, normalized vector field,
    magnitude, magnitude x component and magnitude y component.

    """

    image = spatial_data["images"][0]
    values = spatial_data["derived_quantity"]

    magnitude, x_values, y_values, subtitles = _get_2d_values(values)

    axes = setup_frame_gridspec(
        1, 4, subdivisions_xdir, subdivisions_ydir
    )

    plot_image_subdivision(
        axes[0],
        image,
        magnitude,
        subdivisions_xdir,
        subdivisions_ydir,
    )
    plot_over_time(
        axes[1],
        magnitude,
        time,
        subdivisions_xdir,
        subdivisions_ydir,
    )
    plot_over_time(
        axes[2], x_values, time, subdivisions_xdir, subdivisions_ydir
    )
    plot_over_time(
        axes[3], y_values, time, subdivisions_xdir, subdivisions_ydir
    )

    for i in range(1, 4):
        axes[i][subdivisions_xdir // 2][0].set_ylabel(
            f"{label}, {subtitles[i-1]}"
        )
        axes[i][-1][subdivisions_ydir // 2].set_xlabel("Time (ms)")

    plt.suptitle(label)


def _get_2x2d_values(values):
    xx_values = values[:, :, :, 0, 0]
    xy_values = values[:, :, :, 0, 1]
    sg_values = np.linalg.norm(values, axis=(3, 4))
    yx_values = values[:, :, :, 1, 0]
    yy_values = values[:, :, :, 1, 1]

    subtitles = [
        "component xx",
        "component xy",
        "largest singular value",
        "component yx",
        "component yy",
    ]

    return (
        xx_values,
        xy_values,
        sg_values,
        yx_values,
        yy_values,
        subtitles,
    )


def plot_2x2d_values(
    spatial_data, time, subdivisions_xdir, subdivisions_ydir, label
):
    """

    Plots original image, eigenvalues (max.) and four components of
    a tensor value.

    """

    image = spatial_data["images"][0]
    values = spatial_data["derived_quantity"]

    axes = setup_frame_gridspec(
        2, 3, subdivisions_xdir, subdivisions_ydir
    )

    (
        xx_values,
        xy_values,
        sg_values,
        yx_values,
        yy_values,
        subtitles,
    ) = _get_2x2d_values(values)

    plot_image_subdivision(
        axes[0],
        image,
        sg_values,
        subdivisions_xdir,
        subdivisions_ydir,
    )

    for (axis, value) in zip(
        axes[1:],
        (xx_values, xy_values, sg_values, yx_values, yy_values),
    ):
        plot_over_time(
            axis, value, time, subdivisions_xdir, subdivisions_ydir
        )

    for i in range(1, 6):
        axes[i][subdivisions_xdir // 2][0].set_ylabel(
            f"{label}, {subtitles[i-1]}"
        )
        axes[i][-1][subdivisions_ydir // 2].set_xlabel("Time (ms)")

    plt.suptitle(label)


def _make_filename(
    f_in, metric, subdivisions_x, subdivisions_y, param_list
):
    fname = generate_filename(
        f_in,
        f"{metric}_over_time_and_area_{subdivisions_x}_{subdivisions_y}",
        param_list,
        "",  # png?
        subfolder="mechanics_over_time_and_area",
    )
    fname_png = f"{fname}.png"

    return fname_png


def _plot_over_time(
    spatial_data,
    time,
    subdivisions_xdir,
    subdivisions_ydir,
    label,
    fname,
):

    values = spatial_data["derived_quantity"]

    plot_fn = get_plot_fun(
        values, [plot_1d_values, plot_2d_values, plot_2x2d_values]
    )
    plot_fn(
        spatial_data,
        time,
        subdivisions_xdir,
        subdivisions_ydir,
        label,
    )

    plt.savefig(fname, dpi=300)


def visualize_over_time_and_area(
    f_in, overwrite, overwrite_all, param_list
):
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

    mps_data, mc_data = load_input_data(
        f_in, param_list, overwrite_all
    )

    metrics = param_list[-1].pop("metrics")
    metrics = metrics.split(" ")

    subdivisions_xdir = param_list[-1]["sub_xdir"]
    subdivisions_ydir = param_list[-1]["sub_ydir"]

    for metric in metrics:
        assert (
            metric in mc_data["all_values"].keys()
        ), f"Error: Metric expected to be in {mc_data['all_values'].keys()}"

        print("Making plot for " + metric + " ...")

        fname_png = _make_filename(
            f_in,
            metric,
            subdivisions_xdir,
            subdivisions_ydir,
            param_list,
        )
        metadata, spatial_data, time = setup_for_key(
            mps_data, mc_data, metric
        )
        label = metadata["label"]

        if overwrite or (not os.path.isfile(fname_png)):
            _plot_over_time(
                spatial_data,
                time,
                subdivisions_xdir,
                subdivisions_ydir,
                label,
                fname_png,
            )
            print(
                "Plot at peak done; " + f"image saved to {fname_png}"
            )
        else:
            print(f"Image {fname_png} already exists")
