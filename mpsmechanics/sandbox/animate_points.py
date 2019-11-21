
import sys
import os
import mps
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation
import matplotlib.colors as mcolors

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

    return org_indices[1:]


def get_diamond_stensil(x_i, y_i, X, Y):
    xy_list = []

    distance = 10

    for g_x in range(-distance, distance+1, 2):
        for g_y in range(-distance, distance+1, 2):
            if abs(g_x) + abs(g_y) < distance:
                x = x_i + g_x
                y = y_i + g_y

                if(x >= 0 and x < X and y >= 0 and y < Y):
                    xy_list.append((x, y))
    
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
        
        if y_from < 0:
            diff = -y_from
            y_from += diff

        if x_to > x_max:
            diff = x_from - x_max
            x_from -= diff

        if y_to > y_max:
            diff = y_from - y_max
            y_from -= diff

    else:
        offset = 0
        x_from, y_from = 0, 0
        x_to, y_to = x_max, y_max

    return int(x_from), int(x_to), int(y_from), int(y_to)


def get_filename(input_file, use_max_strain_pt, im_x, im_y, type_filter, sigma):
    folder = os.path.join(input_file[:-4], "mpsmechanics", "point_animation")
    os.makedirs(folder, exist_ok=True)

    if use_max_strain_pt:
        fname = os.path.join(folder, f"point_animation_{type_filter}_{sigma}_max_strain.png")
    else:
        fname = os.path.join(folder, f"point_animation_{type_filter}_{sigma}_{im_x}_{im_y}")

    return fname


def get_mps_data(input_file):
    mps_data = mps.MPS(input_file)

    pixels2um = mps_data.info["um_per_pixel"]
    images = mps_data.frames
    framerate = 0.2*mps_data.framerate
    x_max = mps_data.info["size_x"]
    y_max = mps_data.info["size_y"]

    return pixels2um, images, framerate, x_max, y_max


def get_mpsmechanics_data(input_file, type_filter, sigma):
    source_file = f"analyze_mechanics_{type_filter}_{sigma}"
    source_file = source_file.replace(".", "p")

    mc_data = mc.read_prev_layer(input_file, \
                source_file, mc.analyze_mechanics)

    time = mc_data["time"]
    displacement = mc_data["all_values"]["displacement"]
    pr_strain = mc_data["folded"]["principal_strain"]

    return time, displacement, pr_strain



def animate_points(input_file, type_filter, sigma, use_max_strain_pt=True, im_x=0, im_y=0, offset=100):
    # read from files

    fname = get_filename(input_file, use_max_strain_pt, im_x, im_y, type_filter, sigma)
    pixels2um, images, framerate, x_max, y_max = get_mps_data(input_file)
    time, displacement, pr_strain = get_mpsmechanics_data(input_file, type_filter, sigma)

    displacement /= pixels2um    # scale to be in pixel units

    # coordinates / mappings / etc

    T, X, Y = pr_strain.shape

    x_range = np.linspace(0, x_max, X)
    y_range = np.linspace(0, y_max, Y)
    
    x_i, y_i = get_midpoint(use_max_strain_pt, pr_strain, im_x, im_y, offset)
    x_from, x_to, y_from, y_to = get_frame(x_range, y_range, x_i, y_i, offset)

    xy_list = get_diamond_stensil(x_i, y_i, X, Y)


    x_coords, y_coords = get_xy_coords(xy_list, 0, x_range, y_range, \
            displacement, x_from, y_from)

    # init animation

    fig, axis = plt.subplots(figsize=(5, 5), dpi=300)
    #axis.axis('off')
    
    vmin = np.min(pr_strain)
    vmax = np.max(pr_strain)
     
    clm = cm.get_cmap('Reds')
    norm = mcolors.Normalize(0, 0.3)
    colors = [clm(norm(pr_strain[0, x, y])) for (x, y) in (xy_list)]
 
    im_subplot = axis.imshow(images[x_from:x_to, y_from:y_to, 0], cmap='gray')
    pts_subplot = axis.scatter(y_coords, x_coords, c=colors, s=2)

    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=clm), ticks=[0, 0.1, 0.2, 0.3])
    cbar.ax.set_yticklabels(['0', '0.1', '0.2', '> 0.3'])
    cbar.set_label("Principal strain (magnitude)")

    plt.suptitle("Time: {} ms".format(int(time[0])))

    def update(index):
        colors = [clm(norm(pr_strain[index, x, y])) for (x, y) in (xy_list)]
        x_coords, y_coords = get_xy_coords(xy_list, index, x_range, y_range, \
                displacement, x_from, y_from)
        im_subplot.set_array(images[x_from:x_to, y_from:y_to, index])
        pts_subplot.set_offsets(np.c_[y_coords, x_coords])
        pts_subplot.set_color(colors)

        plt.suptitle("Time: {} ms".format(int(time[index])))

    # perform animation

    writer = animation.writers["ffmpeg"](fps=framerate)
    anim = animation.FuncAnimation(fig, update, T)

    fname = fname.replace(".", "p")
    fname = os.path.splitext(fname)[0]

    anim.save("{}.{}".format(fname, "mp4"), writer=writer)

    plt.close()


def animate_max_strain(input_file, type_filter, sigma):
    animate_points(input_file, type_filter, sigma)

def animate_given_pts(input_file, type_filter, sigma):

    points = [(77, 27)] # MM 4
    for (im_x, im_y) in points:
        animate_points(input_file, type_filter, sigma, im_x=im_x, im_y=im_y, use_max_strain_pt=False)


input_file = sys.argv[1]
type_filter = sys.argv[2]
sigma = float(sys.argv[3])
assert ".nd2" in input_file and "BF" in input_file, "Error: Wrong file formate."

#animate_max_strain(input_file, type_filter, sigma)
animate_given_pts(input_file, type_filter, sigma)
