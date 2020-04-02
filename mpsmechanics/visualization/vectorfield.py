"""

Åshild Telle / Simula Research Laboratory / 2019

"""

import os
import numpy as np
import matplotlib.pyplot as plt

from ..dothemaths.operations import calc_norm_over_time
from .animation_funs import (
    make_animation,
    get_animation_configuration,
)
from .setup_plots import (
    setup_frame,
    load_input_data,
    make_quiver_plot,
    setup_for_key,
    generate_filenames_pngmp4,
)


def _set_ax_units(axis, shape, scale):

    axis.set_aspect("equal")
    num_yticks = 8
    num_xticks = 4

    yticks = [
        int(shape[0] * i / num_yticks) for i in range(num_yticks)
    ]
    ylabels = [
        int(scale * shape[0] * i / num_yticks)
        for i in range(num_yticks)
    ]

    xticks = [
        int(shape[1] * i / num_xticks) for i in range(num_xticks)
    ]
    xlabels = [
        int(scale * shape[1] * i / num_xticks)
        for i in range(num_xticks)
    ]

    axis.set_yticks(yticks)
    axis.set_xticks(xticks)
    axis.set_yticklabels(ylabels)
    axis.set_xticklabels(xlabels)

    axis.set_xlabel(r"$\mu m$")
    axis.set_ylabel(r"$\mu m$")


def _find_arrow_scaling(values, quiver_step):
    return 1.5 / (quiver_step) * np.mean(np.abs(values))


def _find_xy_coords(images, values, quiver_step):
    x_len, y_len = images.shape[1:3]
    x_coords = np.linspace(0, x_len, values.shape[1])
    y_coords = np.linspace(0, y_len, values.shape[2])

    return np.asarray(
        [x_coords[::quiver_step], y_coords[::quiver_step]]
    )


def _plot_vectorfield(spatial_data, time, time_step, metadata):

    values = spatial_data["derived_quantity"]
    images = spatial_data["images"]
    pixels2um = metadata["pixels2um"]

    axes, fig = setup_frame(1, 1, True, True)
    quiver_step = 3

    coords = _find_xy_coords(images, values, quiver_step)

    im_subplot = axes[0].imshow(images[time_step], "gray")
    qu_subplot = make_quiver_plot(
        axes[0],
        values[time_step, ::quiver_step, ::quiver_step],
        coords,
        "red",
        _find_arrow_scaling(values, quiver_step),
    )

    plt.suptitle("Time: {} ms".format(int(time[time_step])))

    _set_ax_units(axes[0], images.shape[1:3], pixels2um)

    def _update(index):
        im_subplot.set_array(images[index])
        qu_subplot.set_UVC(
            values[index, ::quiver_step, ::quiver_step, 1],
            values[index, ::quiver_step, ::quiver_step, 0],
        )

        plt.suptitle("Time: {} ms".format(int(time[index])))

    return fig, _update


def _plot_at_peak(spatial_data, time, metadata, fname):

    values = spatial_data["derived_quantity"]
    peak = np.argmax(calc_norm_over_time(values))

    _plot_vectorfield(spatial_data, time, peak, metadata)

    plt.savefig(fname)
    plt.close("all")


def _make_animation(
    spatial_data, time, metadata, fname, animation_config
):
    fig, _update = _plot_vectorfield(spatial_data, time, 0, metadata)
    make_animation(fig, _update, fname, **animation_config)


def visualize_vectorfield(
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
    animation_config = get_animation_configuration(
        param_list[-1], mps_data
    )
    animate = animation_config.pop("animate")

    metrics = param_list[-1].pop("metrics")
    metrics = metrics.split(" ")

    for metric in metrics:
        assert (
            metric in mc_data["all_values"].keys()
        ), f"Error: Metric expected to be in {mc_data['all_values'].keys()}"

        assert mc_data["all_values"][metric].shape[3:] == (
            2,
        ), f"Error: Vectorfield only defined for vectors."

        print("Plots for " + metric + " ...")

        fname_png, fname_mp4 = generate_filenames_pngmp4(
            f_in, f"vectorfield_{metric}", "visualize_vectorfield", param_list
        )

        metadata, spatial_data, time = setup_for_key(
            mps_data, mc_data, metric
        )

        if overwrite or (not os.path.isfile(fname_png)):
            _plot_at_peak(spatial_data, time, metadata, fname_png)
            print(
                f"Plot at peak done; image saved to {fname_png}"
            )
        else:
            print(f"Image {fname_png} already exists")

        if animate:
            if overwrite or (not os.path.isfile(fname_mp4)):
                _make_animation(
                    spatial_data,
                    time,
                    metadata,
                    fname_mp4,
                    animation_config,
                )
                print("Animation movie produced; movie saved to {fname_mp4}")
            else:
                print(f"Movie {fname_mp4} already exists")

    print("Visualization done, finishing ...")
