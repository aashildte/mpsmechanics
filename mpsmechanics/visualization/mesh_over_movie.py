"""

Åshild Telle / Simula Research Laboratory / 2019

"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..dothemaths.operations import calc_norm_over_time
from ..utils.data_layer import generate_filename
from .animation_funs import (
    get_animation_configuration,
    make_animation,
)
from .setup_plots import load_input_data, generate_filenames_pngmp4


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


def _plot_part_of_image(axis, images, time_step, user_params):
    x_from, x_to, y_from, y_to = _calc_value_range(
        images.shape[1], images.shape[2], user_params
    )
    part_of_im = images[:, x_from:x_to, y_from:y_to]

    im_subplot = axis.imshow(
        part_of_im[time_step], cmap="gray", origin="upper"
    )

    _set_ticks(axis, x_from, x_to, y_from, y_to)

    axis.set_xlim(0, y_to - y_from - 1)
    axis.set_ylim(0, x_to - x_from - 1)

    return im_subplot, part_of_im, [x_from, y_from]


def _calc_mesh_coords(spatial_data, start_indices, step):
    displacement = spatial_data["displacement"]
    images = spatial_data["images"]
    block_size = int(
        np.ceil(images.shape[1] / displacement.shape[1])
    )

    disp_x_dim, disp_y_dim = displacement.shape[1:3]

    x_range = (
        np.linspace(0, disp_x_dim * block_size, disp_x_dim)
        - start_indices[0]
        + block_size / 2
    )
    y_range = (
        np.linspace(0, disp_y_dim * block_size, disp_y_dim)
        - start_indices[1]
        + block_size / 2
    )

    org_ycoords, org_xcoords = np.meshgrid(y_range, x_range)

    all_xcoords = (
        org_xcoords[::step, ::step]
        + displacement[:, ::step, ::step, 0]
    )
    all_ycoords = (
        org_ycoords[::step, ::step]
        + displacement[:, ::step, ::step, 1]
    )

    return all_xcoords, all_ycoords


def _plot_mesh(
    axis, spatial_data, time_step, spatial_step, start_indices
):
    xcoords, ycoords = _calc_mesh_coords(
        spatial_data, start_indices, spatial_step
    )

    all_x_values, all_y_values, all_lines = [], [], []

    for _x in range(xcoords.shape[1]):
        x_values = xcoords[:, _x, :]
        y_values = ycoords[:, _x, :]
        line = axis.plot(
            y_values[time_step],
            x_values[time_step],
            c="white",
            linewidth=0.5,
        )[0]

        all_x_values.append(x_values)
        all_y_values.append(y_values)
        all_lines.append(line)

    for _y in range(ycoords.shape[2]):
        x_values = xcoords[:, :, _y]
        y_values = ycoords[:, :, _y]
        line = axis.plot(
            y_values[time_step],
            x_values[time_step],
            c="white",
            linewidth=0.5,
        )[0]

        all_x_values.append(x_values)
        all_y_values.append(y_values)
        all_lines.append(line)

    return all_x_values, all_y_values, all_lines


def _plot_points(
    axis, spatial_data, time_step, spatial_step, start_indices
):
    xcoords, ycoords = _calc_mesh_coords(
        spatial_data, start_indices, spatial_step
    )
    strain = spatial_data["principal_strain"]

    x_dim, y_dim = strain.shape[1:3]
    x_range = np.arange(0, x_dim, spatial_step)
    y_range = np.arange(0, y_dim, spatial_step)

    x_range = x_range.astype(int)
    y_range = y_range.astype(int)

    pts_subplot = axis.scatter(
        ycoords[time_step].flatten(), xcoords[time_step].flatten()
    )

    return pts_subplot, ycoords, xcoords


def _make_colorbars(fig, axes, clm, norm):

    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cax.remove()

    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=clm), cax=cax
    )

    cbar.set_label("Principal strain (magnitude)")


def _color_points(fig, axes, spatial_data, pts_subplot, time_step):

    strain = spatial_data["principal_strain"]

    clm = cm.get_cmap("Reds")
    norm = mcolors.Normalize(0, np.max(strain))

    colors = clm(
        norm(
            strain.reshape(
                strain.shape[0], strain.shape[1] * strain.shape[2]
            )
        )
    )
    pts_subplot.set_color(colors[time_step])

    _make_colorbars(fig, axes, clm, norm)

    return colors


def _plot_mesh_over_image(
    spatial_data, user_params, time, time_step
):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    image_subplots = []

    for axis in axes:
        im_subplot, im_part, start_indices = _plot_part_of_image(
            axis, spatial_data["images"], time_step, user_params
        )
        image_subplots.append(im_subplot)

    spatial_step = user_params["step"]

    x_values, y_values, lines = _plot_mesh(
        axes[0], spatial_data, time_step, spatial_step, start_indices
    )

    pts_subplot, ycoords, xcoords = _plot_points(
        axes[1], spatial_data, time_step, spatial_step, start_indices
    )

    colors = _color_points(
        fig, axes, spatial_data, pts_subplot, time_step
    )

    plt.suptitle("Time: {} ms".format(int(time[time_step])))

    for axis in axes:
        axis.set_ylim(max(axis.get_ylim()), min(axis.get_ylim()))

    def _update(index):
        for im_subplot in image_subplots:
            im_subplot.set_array(im_part[index])

        for (_x_values, _y_values, line) in zip(
            x_values, y_values, lines
        ):
            line.set_ydata(_x_values[index])
            line.set_xdata(_y_values[index])

        pts_subplot.set_offsets(
            np.c_[ycoords[index].flatten(), xcoords[index].flatten()]
        )
        pts_subplot.set_color(colors[index])

        plt.suptitle("Time: {} ms".format(int(time[index])))

    return fig, _update


def _plot_at_time_step(
    spatial_data, user_params, time, time_step, filename
):
    displacement = spatial_data["displacement"]

    if time_step is None:
        time_step = np.argmax(calc_norm_over_time(displacement))

    _plot_mesh_over_image(spatial_data, user_params, time, time_step)

    plt.savefig(filename)
    plt.close("all")


def _make_animation(
    spatial_data, user_params, time, fname, animation_config
):
    fig, update = _plot_mesh_over_image(
        spatial_data, user_params, time, 0
    )
    make_animation(fig, update, fname, **animation_config)


def _get_image_configuration(params):
    xcoord = params["xcoord"]
    ycoord = params["ycoord"]
    width = params["width"]
    step = params["step"]

    return {
        "step": step,
        "xcoord": xcoord,
        "ycoord": ycoord,
        "width": width,
    }


def _read_input_data(f_in, param_list, overwrite_all):
    mps_data, mc_data = load_input_data(
        f_in, param_list, overwrite_all
    )
    animation_config = get_animation_configuration(
        param_list[-1], mps_data
    )

    displacement_px = (1 / mps_data.info["um_per_pixel"]) * mc_data[
        "all_values"
    ]["displacement"]
    images = np.moveaxis(mps_data.frames, 2, 0)
    time = mc_data["time"]

    spatial_data = {
        "images": images,
        "displacement": displacement_px,
        "principal_strain": mc_data["folded"]["principal_strain"],
    }

    return animation_config, spatial_data, time


def visualize_mesh_over_movie(
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

    time_step = param_list[-1]["time_step"]

    animation_config, spatial_data, time = _read_input_data(
        f_in, param_list, overwrite_all
    )

    fname_png, fname_mp4 = generate_filenames_pngmp4(
            f_in, "mesh_over_images", "mesh_over_images", param_list
    )

    user_params = _get_image_configuration(param_list[-1])

    if overwrite or not os.path.isfile(fname_png):
        _plot_at_time_step(
            spatial_data, user_params, time, time_step, fname_png
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
                spatial_data,
                user_params,
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
