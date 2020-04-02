"""

Åshild Telle / Simula Research Laboratory / 2019

"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

import mps
from ..dothemaths.operations import calc_norm_over_time
from ..utils.data_layer import generate_filename, read_prev_layer
from .animation_funs import (
    get_animation_configuration,
    make_animation,
)
from ..pillar_tracking.pillar_tracking import track_pillars


def _calc_value_range(image_x_dim, image_y_dim, user_params):

    xcoord = user_params["xcoord"]
    ycoord = user_params["ycoord"]
    width = user_params["width"]

    x_from = int(xcoord - width / 2)
    y_from = int(ycoord - width / 2)
    x_to = int(xcoord + width / 2)
    y_to = int(ycoord + width / 2)

    if x_from < 0:
        x_from = 0
    if x_to > image_x_dim:
        x_to = image_x_dim - 1
    if y_from < 0:
        y_from = 0
    if y_to > image_y_dim:
        y_to = image_y_dim - 1

    return x_from, x_to, y_from, y_to


def _set_ticks(axis, x_from, x_to, y_from, y_to):
    axis.set_xlabel("Pixels")
    axis.set_ylabel("Pixels")

    xcoords = np.linspace(0, x_to - x_from - 1, 5)
    x_ticks = np.linspace(x_from, x_to - 1, 5)
    ycoords = np.linspace(0, y_to - y_from - 1, 5)
    y_ticks = np.linspace(y_from, y_to - 1, 5)

    axis.set_xticklabels([int(y) for y in y_ticks])
    axis.set_yticklabels([int(x) for x in x_ticks])
    axis.set_xticks([int(y) for y in ycoords])
    axis.set_yticks([int(x) for x in xcoords])


def _plot_part_of_image(axis, images, time_step, coords):
    width = 150
    x_from = int(coords[0] - width // 2)
    x_to = int(coords[0] + width // 2)
    y_from = int(coords[1] - width // 2)
    y_to = int(coords[1] + width // 2)

    part_of_im = images[:, x_from:x_to, y_from:y_to]

    im_subplot = axis.imshow(
        part_of_im[time_step], cmap="gray", origin="upper"
    )

    _set_ticks(axis, x_from, x_to, y_from, y_to)

    return im_subplot, part_of_im, [x_from, y_from]


def _plot_circle(
    axis, pillar_coords, radius, time_step, start_indices
):
    num_points = 200

    xcoords = (
        pillar_coords[:, 0][:, None]
        + radius * np.cos(np.linspace(0, 2 * np.pi, num_points))
        - start_indices[0]
    )
    ycoords = (
        pillar_coords[:, 1][:, None]
        + radius * np.sin(np.linspace(0, 2 * np.pi, num_points))
        - start_indices[1]
    )

    circle = axis.plot(xcoords[time_step], ycoords[time_step], "r")[
        0
    ]

    return circle, xcoords, ycoords


def _plot_mesh_over_image(
    images, pillar_coords, radius, time, time_step
):

    num_pillars = pillar_coords.shape[1]
    xplots = min(4, num_pillars)
    yplots = int(np.ceil(num_pillars / xplots))
    fig, axes = plt.subplots(
        yplots, xplots, figsize=(4 * xplots, 4 * yplots)
    )

    if xplots > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    image_subplots = []
    im_parts = []
    circle_subplots = []

    for i in range(num_pillars):
        axis = axes[i]
        im_subplot, im_part, start_indices = _plot_part_of_image(
            axis, images, time_step, pillar_coords[time_step, i]
        )

        ci_subplots = _plot_circle(
            axis,
            pillar_coords[:, i],
            radius,
            time_step,
            start_indices,
        )

        image_subplots.append(im_subplot)
        im_parts.append(im_part)
        circle_subplots.append(ci_subplots)

    for i in range(num_pillars, len(axes)):
        axes[i].set_axis_off()

    plt.suptitle("Time: {} ms".format(int(time[time_step])))

    def _update(index):
        for (im_subplot, im_part) in zip(image_subplots, im_parts):
            im_subplot.set_array(im_part[index])

        for i in range(num_pillars):
            circle, xcoords, ycoords = circle_subplots[i]

            circle.set_ydata(xcoords[index])
            circle.set_xdata(ycoords[index])

        plt.suptitle("Time: {} ms".format(int(time[index])))

    return fig, _update


def _plot_at_time_step(
    images, pillar_coords, radius, time, time_step, filename
):
    displacement = pillar_coords

    if time_step is None:
        time_step = np.argmax(
            np.mean(np.linalg.norm(pillar_coords, axis=(2)), axis=1)
        )

    _plot_mesh_over_image(
        images, pillar_coords, radius, time, time_step
    )

    plt.savefig(filename)
    plt.close("all")


def _make_animation(
    images, pillar_coords, radius, time, fname, animation_config
):
    fig, update = _plot_mesh_over_image(
        images, pillar_coords, radius, time, 0
    )
    make_animation(fig, update, fname, **animation_config)


def _generate_param_filename(f_in, user_params):
    fname = generate_filename(
        f_in,
        f"pillars",
        user_params,
        "",
        subfolder="pillar_tracking",  # mp3 or png
    )
    return fname


def _read_input_data(f_in, param_list, overwrite_all):
    mps_data = mps.MPS(f_in)

    pillar_disp = read_prev_layer(
        f_in, track_pillars, param_list[:-1], overwrite_all
    )
    animation_config = get_animation_configuration(
        param_list[-1], mps_data
    )

    pillar_coords = (
        pillar_disp["displacement_pixels"]
        + pillar_disp["initial_positions"]
    )
    radius = (
        pillar_disp["material_parameters"]["R"]
        / mps_data.info["um_per_pixel"]
    )
    images = np.moveaxis(mps_data.frames, 2, 0)

    time = mps_data.time_stamps

    return animation_config, images, pillar_coords, radius, time


def visualize_pillar_tracking(
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

    (
        animation_config,
        images,
        pillar_coords,
        radius,
        time,
    ) = _read_input_data(f_in, param_list, overwrite_all)

    fname_p = _generate_param_filename(f_in, param_list)
    fname_png = fname_p + ".png"
    fname_mp4 = fname_p + ".mp4"
    time_step = param_list[-1]["time_step"]

    if overwrite or not os.path.isfile(fname_png):
        _plot_at_time_step(
            images, pillar_coords, radius, time, time_step, fname_png
        )
        print(
            "Plots of mesh over image at peak done; "
            + f"image saved to {fname_png}"
        )
    else:
        print(f"Image {fname_png} already exist.")

    animate = animation_config.pop("animate")

    if animate:
        if overwrite or not os.path.isfile(fname_mp4):
            print("Making a movie ..")
            _make_animation(
                images,
                pillar_coords,
                radius,
                time,
                fname_mp4,
                animation_config,
            )

            print(
                "Movie of mesh over image produced; "
                + f"movie saved to {fname_mp4}"
            )
        else:
            print(f"Movie {fname_mp4} already exist.")
