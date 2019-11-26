
import os
import mps
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

import mpsmechanics as mc


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


def _mesh_over_image(org_x_coords, org_y_coords, coord_range, step, displacement, images, time, time_step):
    x_coords = org_x_coords + displacement[time_step, :, :, 0]
    y_coords = org_y_coords + displacement[time_step, :, :, 1]
    
    x_from, x_to, y_from, y_to = coord_range

    fig = plt.figure()
    axis = fig.add_subplot()
    
    _set_ticks(axis, x_from, x_to, y_from, y_to)
    axis.set_xlim(0, y_to-y_from-1)
    axis.set_ylim(0, x_to-x_from-1)

    im_subplot = axis.imshow(images[x_from:x_to, y_from:y_to, 0], cmap='gray')

    T, X, Y, _ = displacement.shape

    all_lines = []

    for x in range(0, X, step):
        x_values = x_coords[x, :]
        y_values = y_coords[x, :]
        all_lines.append(axis.plot(y_values, x_values, c='white', linewidth=0.5)[0])

    for y in range(0, Y, step):
        x_values = x_coords[:, y]
        y_values = y_coords[:, y]
        all_lines.append(axis.plot(y_values, x_values, c='white', linewidth=0.5)[0])
    plt.suptitle("Time: {} ms".format(int(time[time_step])))
    
    return fig, axis, im_subplot, all_lines


def _calc_value_range(x_coord, y_coord, width, X_image, Y_image):
    x_from = int(x_coord - width/2)
    y_from = int(y_coord - width/2)
    x_to = int(x_coord + width/2)
    y_to = int(y_coord + width/2)

    if x_from < 0:
        x_from = 0
    if x_to > X_image:
        x_to = X_image - 1
    if y_from < 0:
        y_from = 0
    if y_to > Y_image:
        y_to = Y_image - 1

    return x_from, x_to, y_from, y_to


def _mesh_over_movie(mps_data, mc_data, animate, scaling_factor, width, x_coord, y_coord, step, fname):

    displacement = mc_data["all_values"]["displacement"]
    T, X, Y, _ = displacement.shape
    time = mc_data["time"]
    peak = np.argmax(mc_data["over_time_avg"]["displacement"])

    images = mps_data.frames 
    framerate = mps_data.framerate
    
    X_image, Y_image = images.shape[:2]
    X_disp, Y_disp = displacement.shape[1:3]
   
    x_from, x_to, y_from, y_to = _calc_value_range(x_coord, y_coord, width, X_image, Y_image)

    x_range = np.linspace(0, X_disp * (X_image // X_disp), X_disp)
    y_range = np.linspace(0, Y_disp * (Y_image // Y_disp), Y_disp)

    org_y_coords, org_x_coords = np.meshgrid(y_range - y_from, x_range - x_from)

    # one plot
    _mesh_over_image(org_x_coords, org_y_coords, \
            [x_from, x_to, y_from, y_to], step, displacement, images, time, peak)
    plt.savefig(fname + ".png")

    # and one movie, if applicable

    if animate:

        fig, axis, im_subplot, all_lines = _mesh_over_image(org_x_coords, org_y_coords, \
                [x_from, x_to, y_from, y_to], step, displacement, images, time, 0)

        def update(index):
            x_coords = org_x_coords + displacement[index, :, :, 0]
            y_coords = org_y_coords + displacement[index, :, :, 1]

            #im_subplot.set_array(images[:, :, index])
            im_subplot.set_array(images[x_from:x_to, y_from:y_to, index])

            cnt = 0
            for x in range(0, X, step):
                x_values = x_coords[x, :]
                y_values = y_coords[x, :]
                all_lines[cnt].set_ydata(x_values)
                all_lines[cnt].set_xdata(y_values)
                cnt += 1

            for y in range(0, Y, step):
                x_values = x_coords[:, y]
                y_values = y_coords[:, y]
                all_lines[cnt].set_ydata(x_values)
                all_lines[cnt].set_xdata(y_values)
                cnt += 1 

            plt.suptitle("Time: {} ms".format(int(time[index])))
        
        # perform animation
        writer = animation.writers["ffmpeg"](fps=scaling_factor*framerate)
        anim = animation.FuncAnimation(fig, update, T)

        fname = os.path.splitext(fname)[0]
        anim.save("{}.{}".format(fname, "mp4"), writer=writer)

        plt.close()

    

def animate_mesh_over_movie(f_in, overwrite, param_list):
    """

    "main function"

    """
    
    print("Parameters animation of mesh over movie:")
    for key in param_list[2].keys():
        print(" * {}: {}".format(key, param_list[2][key]))
    
    mps_data = mps.MPS(f_in)
    mc_data = mc.read_prev_layer(
        f_in,
        mc.analyze_mechanics,
        param_list[:-1],
        overwrite
    )

    x_coord = param_list[2]["x_coord"]
    y_coord = param_list[2]["y_coord"]
    width = param_list[2]["width"]

    output_folder = mc.make_dir_layer_structure(f_in, \
            os.path.join("mpsmechanics", "animate_mesh_over_movie"))
    os.makedirs(output_folder, exist_ok=True)

    fname = mc.generate_filename(f_in, \
                                 os.path.join("animate_mesh_over_movie", f"animation_{x_coord}_{y_coord}_{width}"), \
                                 param_list[:2])

    _mesh_over_movie(mps_data, mc_data, fname=fname, **param_list[-1])

    print("Visualization done, finishing ..")
