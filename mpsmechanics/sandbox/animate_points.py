
import os
import mps
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation

import mpsmechanics as mc


def get_xy_coords(xy_list, index, x_range, y_range, displacement, offset_x, offset_y):
    x_coords = []
    y_coords = []

    for (x, y) in xy_list:
        coord_indices = (index, x, y)
        x_ind = coord_indices + (0,)
        y_ind = coord_indices + (1,)

        x_coords.append(x_range[coord_indices[1]] + displacement[x_ind] - offset_x)
        y_coords.append(y_range[coord_indices[2]] + displacement[y_ind] - offset_y)

    return x_coords, y_coords


def get_midpoint(use_max_strain_pt, pr_strain, im_x, im_y, offset):
    if use_max_strain_pt:
        org_indices = np.unravel_index(np.argmax(pr_strain), pr_strain.shape)
    else:
        org_indices = (0, im_x, im_y)

    x_i = org_indices[1] + im_x
    y_i = org_indices[2] + im_y

    return x_i, y_i


def get_5pt_stensil(x_i, y_i, X, Y):
    xy_list = [(x_i, y_i)]

    if y_i > 0:
        xy_list.append((x_i, y_i-1))
    if x_i > 0:
        xy_list.append((x_i-1, y_i))
    if y_i < Y - 1:
        xy_list.append((x_i, y_i+1))
    if x_i < X-1:
        xy_list.append((x_i+1, y_i))

    return xy_list


def get_frame(x_range, y_range, x_i, y_i, offset=0):

    x_max = x_range[-1]
    y_max = y_range[-1]

    if offset != 0:
        x_from, y_from = x_range[x_i] - offset, y_range[y_i] - offset
        x_to, y_to = x_range[x_i] + offset, y_range[y_i] + offset

        if x_from < 0:
            diff = -x_from
            x_from += diff
            x_to += diff

        if x_from < 0:
            diff = -y_from
            y_from += diff
            y_to += diff

        if x_to > x_max:
            diff = x_from - x_max
            x_from -= diff
            x_to -= diff

        if y_to > y_max:
            diff = y_from - y_max
            y_from -= diff
            y_to -= diff
    else:
        offset = 0
        x_from, y_from = 0, 0
        x_to, y_to = x_max, y_max

    return int(x_from), int(x_to), int(y_from), int(y_to)


def get_filename(input_file, use_max_strain_pt, im_x, im_y):
    folder = os.path.join(input_file[:-4], "mpsmechanics", "point_animation")
    os.makedirs(folder, exist_ok=True)

    if use_max_strain_pt:
        fname = os.path.join(folder, "point_animation_max_strain.png")
    else:
        fname = os.path.join(folder, f"point_animation_{im_x}_{im_y}")

    return fname


def get_mps_data(input_file):
    mps_data = mps.MPS(input_file)

    pixels2um = mps_data.info["um_per_pixel"]
    images = mps_data.frames
    framerate = 0.2*mps_data.framerate
    x_max = mps_data.info["size_x"]
    y_max = mps_data.info["size_y"]

    return pixels2um, images, framerate, x_max, y_max


def get_mpsmechanics_data(input_file):
    file_pref = input_file[:-4]
    
    mc_data = read_prev_layer(input_file, \
                "analyze_mechanics", analyze_mechanics)

    time = mc_data["time"]
    displacement = mc_data["all_values"]["displacement"]
    pr_strain = mc_data["folded"]["principal_strain"]

    return time, displacement, pr_strain



def animate_points(input_file, use_max_strain_pt=True, im_x=0, im_y=0, offset=100):

    # read from files

    fname = get_filename(input_file, use_max_strain_pt, im_x, im_y)
    pixels2um, images, framerate, x_max, y_max = get_mps_data(input_file)
    time, displacement, pr_strain = get_mpsmechanics_data(input_file)

    displacement /= pixels2um    # scale to be in pixel units

    # coordinates / mappings / etc

    T, X, Y = pr_strain.shape

    x_range = np.linspace(0, x_max, X)
    y_range = np.linspace(0, y_max, Y)

    x_i, y_i = get_midpoint(use_max_strain_pt, pr_strain, im_x, im_y, offset)
    x_from, x_to, y_from, y_to = get_frame(x_range, y_range, x_i, y_i, offset)

    xy_list = get_5pt_stensil(x_i, y_i, X, Y)
    clm = cm.get_cmap('Wistia')
    colors = [clm(0.2*x) for x in range(len(xy_list))]

    x_coords, y_coords = get_xy_coords(xy_list, 0, x_range, y_range, \
            displacement, x_from, y_from)

    # init animation

    fig, axis = plt.subplots(figsize=(5, 5), dpi=300)
    axis.axis('off')
    
    im_subplot = axis.imshow(images[x_from:x_to, y_from:y_to, 0], \
            cmap=cm.get_cmap('gray'))
    pts_subplot = axis.scatter(y_coords, x_coords, c=colors)

    plt.suptitle("Time: {} ms".format(int(time[0])))

    def update(index):
        x_coords, y_coords = get_xy_coords(xy_list, index, x_range, y_range, \
                displacement, x_from, y_from, pixels2um)
        im_subplot.set_array(images[x_from:x_to, y_from:y_to, index])
        pts_subplot.set_offsets(np.c_[y_coords, x_coords])

        plt.suptitle("Time: {} ms".format(int(time[index])))

    # perform animation

    writer = animation.writers["ffmpeg"](fps=framerate)
    anim = animation.FuncAnimation(fig, update, T)

    fname = os.path.splitext(fname)[0]
    anim.save("{}.{}".format(fname, "mp4"), writer=writer)

    plt.close()


def animate_max_strain(input_file):
    animate_points(input_file)

def animate_uniform_pts(input_file):
    # Vet ikke om dette er beste fordeling?

    for im_x in range(35, 200, 40):
        for im_y in range(25, 55, 4):
            animate_points(input_file, im_x=im_x, im_y=im_y, use_max_strain_pt=False)

try:
    input_file = sys.argv[1]

    assert ".nd2" in f_in and "BF" in f_in, "Error: Wrong file formate."
except:
    print("Expected first argument: BF nd2 file")

animate_max_strain(input_file)
animate_uniform_pts(input_file)
