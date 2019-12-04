"""

Åshild Telle / Simula Research Laboratory / 2019

"""
import os
import numpy as np
import matplotlib.pyplot as plt

from ..dothemaths.operations import calc_norm_over_time
from ..utils.data_layer import generate_filename
from .animation_funs import get_animation_configuration, make_animation
from .setup_plots import load_input_data

def _calc_value_range(image_x_dim, image_y_dim, x_coord, y_coord, width):
    x_from = int(x_coord - width/2)
    y_from = int(y_coord - width/2)
    x_to = int(x_coord + width/2)
    y_to = int(y_coord + width/2)

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

    x_coords = np.linspace(0, x_to-x_from-1, 5)
    x_ticks = np.linspace(x_from, x_to-1, 5)
    y_coords = np.linspace(0, y_to-y_from-1, 5)
    y_ticks = np.linspace(y_from, y_to-1, 5)

    axis.set_xticklabels([int(y) for y in y_ticks])
    axis.set_yticklabels([int(x) for x in x_ticks])
    axis.set_xticks([int(y) for y in y_coords])
    axis.set_yticks([int(x) for x in x_coords])


def _plot_part_of_image(axis, images, time_step, im_config):
    x_from, x_to, y_from, y_to = _calc_value_range(images.shape[1], images.shape[2], **im_config)
    part_of_im = images[:, x_from:x_to, y_from:y_to]
    im_subplot = plt.imshow(part_of_im[time_step], cmap='gray')

    _set_ticks(axis, x_from, x_to, y_from, y_to)

    plt.xlim(0, y_to-y_from-1)
    plt.ylim(0, x_to-x_from-1)

    return im_subplot, part_of_im, [x_from, y_from]


def _calc_mesh_coords(spatial_data, start_indices, step):
    displacement = spatial_data["displacement"]
    images = spatial_data["images"]
    block_size = images.shape[1] // displacement.shape[1]

    disp_x_dim, disp_y_dim = displacement.shape[1:3]

    x_range = np.linspace(0, disp_x_dim*block_size, \
                          disp_x_dim // step) - start_indices[0]
    y_range = np.linspace(0, disp_y_dim*block_size, \
                          disp_y_dim // step) - start_indices[1]

    org_y_coords, org_x_coords = np.meshgrid(y_range, x_range)

    all_x_coords = org_x_coords + displacement[:, ::step, ::step, 0]
    all_y_coords = org_y_coords + displacement[:, ::step, ::step, 1]

    return all_x_coords, all_y_coords


def _plot_mesh(spatial_data, time_step, spatial_step, start_indices):
    x_coords, y_coords = _calc_mesh_coords(spatial_data, start_indices, \
            spatial_step)

    all_x_values, all_y_values, all_lines = [], [], []

    for _x in range(x_coords.shape[1]):
        x_values = x_coords[:, _x, :]
        y_values = y_coords[:, _x, :]
        line = plt.plot(y_values[time_step], \
                        x_values[time_step], \
                        c='white', linewidth=0.5)[0]

        all_x_values.append(x_values)
        all_y_values.append(y_values)
        all_lines.append(line)

    for _y in range(y_coords.shape[2]):
        x_values = x_coords[:, :, _y]
        y_values = y_coords[:, :, _y]
        line = plt.plot(y_values[time_step], \
                        x_values[time_step], \
                        c='white', linewidth=0.5)[0]

        all_x_values.append(x_values)
        all_y_values.append(y_values)
        all_lines.append(line)

    return all_x_values, all_y_values, all_lines


def _plot_mesh_over_image(spatial_data, user_params, time, time_step):
    fig = plt.figure()
    axis = fig.add_subplot()

    im_subplot, im_part, start_indices = \
            _plot_part_of_image(axis, \
                                spatial_data["images"], \
                                time_step, \
                                user_params["im_config"])

    x_values, y_values, lines = \
            _plot_mesh(spatial_data, time_step, \
                       user_params["step"], start_indices)

    plt.suptitle("Time: {} ms".format(int(time[time_step])))

    def _update(index):
        im_subplot.set_array(im_part[index])
        for (_x_values, _y_values, line) in zip(x_values, y_values, lines):
            line.set_ydata(_x_values[index])
            line.set_xdata(_y_values[index])

        plt.suptitle("Time: {} ms".format(int(time[index])))

    return fig, _update


def _plot_at_peak(spatial_data, user_params, time, fname):
    displacement = spatial_data["displacement"]

    peak = np.argmax(calc_norm_over_time(displacement))
    _plot_mesh_over_image(spatial_data, user_params, time, peak)

    filename = fname + ".png"
    plt.savefig(filename)
    plt.close('all')


def _make_animation(spatial_data, user_params, time, fname, animation_config):
    fig, update = _plot_mesh_over_image(spatial_data, user_params, time, 0)
    make_animation(fig, update, fname, **animation_config)


def _get_image_configuration(params):
    x_coord = params["x_coord"]
    y_coord = params["y_coord"]
    width = params["width"]
    step = params["step"]

    return {"step" : step,
            "im_config" : {"x_coord" : x_coord,
                           "y_coord" : y_coord,
                           "width" : width}}


def _generate_param_filename(f_in, param_list, user_params):
    x_coord = user_params["im_config"]["x_coord"]
    y_coord = user_params["im_config"]["y_coord"]
    width = user_params["im_config"]["width"]
    step = user_params["step"]

    fname = generate_filename(f_in, \
                              f"mesh_over_images_{x_coord}_{y_coord}_{width}_{step}",
                              param_list[:2],
                              "",        # mp3 or png
                              subfolder="mesh_over_images")
    return fname


def animate_mesh_over_movie(f_in, overwrite, overwrite_all, param_list):
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

    print("Parameters visualize distributions:")

    for key in param_list[2].keys():
        print(" * {}: {}".format(key, param_list[2][key]))

    mps_data, mc_data = load_input_data(f_in, param_list, overwrite_all)
    animation_config = get_animation_configuration(param_list[2], mps_data)

    displacement = mc_data["all_values"]["displacement"]
    images = np.moveaxis(mps_data.frames, 2, 0)
    time = mc_data["time"]

    spatial_data = {"images" : images,
                    "displacement" : displacement}
    user_params = _get_image_configuration(param_list[-1])
    fname = _generate_param_filename(f_in, param_list, user_params)

    if overwrite or not os.path.isfile(fname + ".png"):
        _plot_at_peak(spatial_data, user_params, time, fname)

    animate = animation_config.pop("animate")

    if animate and (overwrite or not os.path.isfile(fname + ".mp4")):
        print("Making a movie ..")
        _make_animation(spatial_data, user_params, time, fname, animation_config)

    print("Mesh over movie done, finishing ...")
