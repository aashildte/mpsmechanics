
import os
import mps
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

def get_diamond_stensil(im_x, im_y, X, Y):
    xy_list = []

    distance = 3

    for g_x in range(-distance, distance+1):
        for g_y in range(-distance, distance+1):
            if abs(g_x) + abs(g_y) < distance:
                x = im_x + g_x
                y = im_y + g_y

                if(x >= 0 and x < X and y >= 0 and y < Y):
                    xy_list.append((x, y))
    xy_list.sort()

    return xy_list


def get_frame(x_range, y_range, im_x, im_y, offset=0):


    x_max = x_range[-1]
    y_max = y_range[-1]


    if offset != 0:
        x_from, y_from = x_range[im_x] - offset, y_range[im_y] - offset
        x_to, y_to = x_range[im_x] + offset, y_range[im_y] + offset

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


def get_filename(input_file, matching_method, block_size, type_filter, sigma, im_x_um, im_y_um, offset, type_animation="mesh"):
    if type_animation=="points":
        folder = os.path.join(input_file[:-4], "mpsmechanics", "point_animation")
    else:
        folder = os.path.join(input_file[:-4], "mpsmechanics", "mesh_animation")

    os.makedirs(folder, exist_ok=True)

    im_x = int(im_x_um)
    im_y = int(im_x_um)
    fname = os.path.join(folder, f"point_animation_{matching_method}_{block_size}_{type_filter}_{sigma}_{im_x}_{im_y}_{offset}")
    return fname

def get_mps_data(input_file):
    mps_data = mps.MPS(input_file)

    pixels2um = mps_data.info["um_per_pixel"]
    images = mps_data.frames
    framerate = mps_data.framerate
    x_max = mps_data.info["size_x"]
    y_max = mps_data.info["size_y"]

    return pixels2um, images, framerate, x_max, y_max


def get_mpsmechanics_data(input_file, matching_method, block_size, type_filter, sigma):
    source_file = f"analyze_mechanics_{matching_method}_{block_size}_{type_filter}_{sigma}"
    source_file = source_file.replace(".", "p")

    kwargs = {"matching_method" : matching_method, "block_size" : block_size, "type_filter" : type_filter, "sigma" : sigma}

    mc_data = mc.read_prev_layer(input_file, \
                source_file, mc.analyze_mechanics, kwargs)

    time = mc_data["time"]
    displacement = mc_data["all_values"]["displacement"]
    pr_strain = mc_data["folded"]["principal_strain"] 
    
    max_strain = np.unravel_index(np.argmax(pr_strain), pr_strain.shape)
    max_disp = np.unravel_index(np.argmax(displacement), displacement.shape)
    print("max strain, disp indices: ", max_strain, max_disp)

    return time, displacement, pr_strain

def set_ticks(axis, x_from, y_from, x_to, y_to)
    axis.set_xlabel("Pixels")
    axis.set_ylabel("Pixels")

    x_coords = np.linspace(0, x_to-x_from-1, 5)
    x_ticks = np.linspace(x_from, x_to-1, 5)
    y_coords = np.linspace(0, y_to-y_from-1, 5)
    y_ticks = np.linspace(y_from, y_to-1, 5)

    axis.set_yticklabels([int(y) for y in y_ticks])
    axis.set_xticklabels([int(x) for x in x_ticks])
    axis.set_yticks([int(y) for y in y_coords])
    axis.set_xticks([int(x) for x in x_coords])


def animate_mesh_over_image(pr_strain, displacement, images, x_from, x_to, y_from, y_to, time, framerate, x_range, y_range, fname):
    step_size = 1

    scale = displacement.shape[1] / images.shape[0]
    block_size = images.shape[0] // displacement.shape[1]
     
    print("x range: ", x_range)
    print(x_from, x_to, y_from, y_to)

    org_y_coords, org_x_coords = np.meshgrid(y_range - y_from, x_range - x_from)

    x_coords = org_x_coords + displacement[0, :, :, 0]
    y_coords = org_y_coords + displacement[0, :, :, 1]

    fig = plt.figure()
    axis = fig.add_subplot()

    set_ticks(axis, x_from, x_to, y_from, y_to)
    axis.set_xlim(0, x_to-x_from-1)
    axis.set_ylim(0, y_to-y_from-1)

    im_subplot = axis.imshow(images[x_from:x_to, y_from:y_to, 0], cmap='gray')
    #im_subplot = axis.imshow(images[:, :, 0], cmap='gray')

    T, X, Y, _ = displacement.shape

    all_values = []

    for x in range(0, X, step_size):
        x_values = x_coords[x, :]
        y_values = y_coords[x, :]
        all_values.append(axis.plot(y_values, x_values, c='white', linewidth=0.5)[0])

    for y in range(0, Y, step_size):
        x_values = x_coords[:, y]
        y_values = y_coords[:, y]
        all_values.append(axis.plot(y_values, x_values, c='white', linewidth=0.5)[0])
    plt.suptitle("Time: {} ms".format(int(time[0])))
    
    def update(index):
        x_coords = org_x_coords + displacement[index, :, :, 0]
        y_coords = org_y_coords + displacement[index, :, :, 1]
    
        #im_subplot.set_array(images[:, :, index])
        im_subplot.set_array(images[x_from:x_to, y_from:y_to, index])

        cnt = 0
        for x in range(0, X, step_size):
            x_values = x_coords[x, :]
            y_values = y_coords[x, :]
            all_values[cnt].set_ydata(x_values)
            all_values[cnt].set_xdata(y_values)
            cnt += 1

        for y in range(0, Y, step_size):
            x_values = x_coords[:, y]
            y_values = y_coords[:, y]
            all_values[cnt].set_ydata(x_values)
            all_values[cnt].set_xdata(y_values)
            cnt += 1 

        plt.suptitle("Time: {} ms".format(int(time[index])))

    #plt.tight_layout()

    # perform animation
    writer = animation.writers["ffmpeg"](fps=0.2*framerate)
    anim = animation.FuncAnimation(fig, update, T)

    fname = os.path.splitext(fname)[0]
    anim.save("{}.{}".format(fname, "mp4"), writer=writer)

    plt.close()


def animate_points_over_image(pr_strain, displacement, xy_list, images, x_from, x_to, y_from, y_to, x_range, y_range, time, framerate, fname):
    
    x_coords, y_coords = get_xy_coords(xy_list, 0, x_range, y_range, \
            displacement, x_from, y_from)

    fig, axis = plt.subplots(figsize=(5, 5), dpi=300)
    #axis.axis('off')

    vmin = np.min(pr_strain)
    vmax = np.max(pr_strain)
     
    clm = cm.get_cmap('Reds')
    norm = mcolors.Normalize(0, 0.2)
    colors = [clm(norm(pr_strain[0, x, y])) for (x, y) in (xy_list)]
 
    im_subplot = axis.imshow(images[x_from:x_to, y_from:y_to, 0], cmap='gray')
    pts_subplot = axis.scatter(y_coords, x_coords, c=colors, s=2)

    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=clm), \
            ticks=[0, 0.1, 0.2], cax=cax)
    cbar.ax.set_yticklabels(['0', '0.1', '> 0.2'])
    cbar.set_label("Principal strain (magnitude)")

    plt.suptitle("Time: {} ms".format(int(time[0])))
    
    set_ticks(axis, x_from, x_to, y_from, y_to)

    def update(index):
        colors = [clm(norm(pr_strain[index, x, y])) for (x, y) in (xy_list)]
        x_coords, y_coords = get_xy_coords(xy_list, index, x_range, y_range, \
                displacement, x_from, y_from)
        im_subplot.set_array(images[x_from:x_to, y_from:y_to, index])
        pts_subplot.set_offsets(np.c_[y_coords, x_coords])
        pts_subplot.set_color(colors)

        plt.suptitle("Time: {} ms".format(int(time[index])))

    plt.tight_layout()

    # perform animation
    T = len(pr_strain)
    writer = animation.writers["ffmpeg"](fps=0.1*framerate)
    anim = animation.FuncAnimation(fig, update, T)

    fname = os.path.splitext(fname)[0]

    anim.save("{}.{}".format(fname, "mp4"), writer=writer)

    plt.close()


def animate_points(input_file, matching_method, block_size, type_filter, sigma, \
        im_x_um=0, im_y_um=0, offset=0):
    # read from files

    fname = get_filename(input_file, matching_method, block_size, type_filter, sigma, im_x_um, im_y_um, offset)
    pixels2um, images, framerate, x_max, y_max = get_mps_data(input_file)
    time, displacement, pr_strain = get_mpsmechanics_data(input_file, \
            matching_method, block_size, type_filter, sigma)

    displacement /= pixels2um    # scale to be in pixel units

    # coordinates / mappings / etc

    T, X, Y = pr_strain.shape

    x_range = np.linspace(0, x_max, X)
    y_range = np.linspace(0, y_max, Y)
   
    im_x = 0
    im_y = 0

    while x_range[im_x] < im_x_um:
        im_x +=1

    while y_range[im_y] < im_y_um:
        im_y +=1

    x_from, x_to, y_from, y_to = get_frame(x_range, y_range, im_x, im_y, offset)
    xy_list = get_diamond_stensil(im_x, im_y, X, Y)

    #animate_points_over_image(pr_strain, displacement, xy_list, images, x_from, x_to, y_from, y_to, x_range, y_range, time, framerate, fname + "_over_image")
    animate_mesh_over_image(pr_strain, displacement, images, x_from, x_to, y_from, y_to, time, framerate, x_range, y_range, fname + "_with_mesh")



