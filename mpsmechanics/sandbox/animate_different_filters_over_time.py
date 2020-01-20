
import os
import sys
import matplotlib.pyplot as plt
from matplotlib import animation, cm
import numpy as np

import mpsmechanics as mc
import mps
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_y_coord(images, y_coord, blocksize, fname):

    plt.imshow(images[:, :, 0], cmap=cm.gray)
    x_coords = [0, images.shape[0]]
    y_coords = [y_coord*blocksize + blocksize//2]*2
    plt.plot(y_coords, x_coords, 'r')
    plt.axis('off')
    plt.savefig(fname + ".png", dpi=300)
    plt.clf()

def get_filename(input_file, quantity):
    folder = os.path.join(input_file[:-4], "mpsmechanics", "animation_over_time")
    os.makedirs(folder, exist_ok=True)

    fname = os.path.join(folder, quantity)

    return fname


def _make_animation(fig, update, framerate, num_time_steps, fname):
    writer = animation.writers["ffmpeg"](fps=framerate)
    anim = animation.FuncAnimation(fig, update, num_time_steps)

    fname = os.path.splitext(fname)[0]
    anim.save("{}.{}".format(fname, "mp4"), dpi=300, writer=writer)

    plt.close()


def _setup_2plots(data, sigmas, key, x_lim, y_coord, time_step):
    fig, axes = plt.subplots(2, 1, figsize=(15, 7), \
            sharex=True, sharey=True)
    axes.flatten()

    x_values = []
    y_values = []

    for data_s in data:
        qt_values = data_s["all_values"][key]
        x_values.append(qt_values[:, :, y_coord, 0])
        y_values.append(qt_values[:, :, y_coord, 1])

    x_values = np.array(x_values)
    y_values = np.array(y_values)

    time = data_s["time"]
    label = mc.make_pretty_label(key, data[0]["unit"][key])
    
    x_range = np.linspace(0, x_lim, x_values.shape[2])

    vmax = 1.1*np.max(x_values)
    
    for axis in axes:
        axis.set_xlim(0, x_range[-1])
        axis.set_ylim(-vmax, vmax)
    
    axes[0].set_title("X component")
    axes[1].set_title("Y component")
    axes[1].set_xlabel(r"X coordinates ($\mu m$)")
   
    plt.suptitle(f"{label}\nTime: {int(time[0])} ms")

    x_lines = []
    y_lines = []    

    for (i, sigma) in enumerate(sigmas):
        linex, = axes[0].plot(x_range, x_values[i][time_step], alpha=0.7, label=fr"$\sigma$ : {sigma}")
        liney, = axes[1].plot(x_range, y_values[i][time_step], alpha=0.7)
        x_lines.append(linex)
        y_lines.append(liney)
    
    axes[0].legend(loc='center left', bbox_to_anchor=(1.05, 0.5))

    def update(index):
        for (i, line) in enumerate(x_lines):
            line.set_ydata(x_values[i][index])
        for (i, line) in enumerate(y_lines):
            line.set_ydata(y_values[i][index])
        
        plt.suptitle(f"{label}\nTime: {int(time[index])} ms")

    fig.subplots_adjust(top=0.85)

    return fig, update, len(time)



def _setup_2x2plots(data, sigmas, key, x_lim, shift, y_coord, time_step):
    fig, axes = plt.subplots(2, 2, figsize=(15, 7), \
            sharex=True)
    axes = axes.flatten()
    
    ux_values = []
    uy_values = []
    vx_values = []
    vy_values = []

    for data_s in data:
        qt_values = data_s["all_values"][key]
        ux_values.append(qt_values[:, :, y_coord, 0, 0])
        uy_values.append(qt_values[:, :, y_coord, 0, 1])
        vx_values.append(qt_values[:, :, y_coord, 1, 0])
        vy_values.append(qt_values[:, :, y_coord, 1, 1])

    ux_values = np.array(ux_values)
    uy_values = np.array(uy_values)
    vx_values = np.array(vx_values)
    vy_values = np.array(vy_values)

    time = data_s["time"]
    label = mc.make_pretty_label(key, data[0]["unit"][key])
    
    x_range = np.linspace(0, x_lim, ux_values.shape[2])

    vmax = 1.0*max([np.max(ux_values - shift), np.max(uy_values), \
                   np.max(vx_values), np.max(vy_values - shift)])

    for axis in [axes[0], axes[3]]:
        axis.set_xlim(0, x_range[-1])
        axis.set_ylim(-vmax + shift, vmax + shift)

    for axis in [axes[1], axes[2]]:
        axis.set_xlim(0, x_range[-1])
        axis.set_ylim(-vmax, vmax)
    

    axes[0].set_title(r"$u_x$")
    axes[1].set_title(r"$u_y$")
    axes[2].set_title(r"$v_x$")
    axes[3].set_title(r"$v_y$")

    axes[2].set_xlabel(r"X coordinates ($\mu m$)")
    axes[3].set_xlabel(r"X coordinates ($\mu m$)")
   
    plt.suptitle(f"{label}\nTime: {int(time[0])} ms")

    ux_lines = []
    uy_lines = []    
    vx_lines = []
    vy_lines = []    

    for (i, sigma) in enumerate(sigmas):
        lineux, = axes[0].plot(x_range, ux_values[i][time_step], alpha=0.7)
        lineuy, = axes[1].plot(x_range, uy_values[i][time_step], alpha=0.7, label=fr"$\sigma$ : {sigma}")
        linevx, = axes[2].plot(x_range, vx_values[i][time_step], alpha=0.7)
        linevy, = axes[3].plot(x_range, vy_values[i][time_step], alpha=0.7)

        for (_elem, _list) in zip([lineux, lineuy, linevx, linevy], [ux_lines, uy_lines, vx_lines, vy_lines]):
            _list.append(_elem)

    axes[1].legend(loc='center left', bbox_to_anchor=(1.05, 0.5))

    def update(index):
        for (i, line) in enumerate(ux_lines):
            line.set_ydata(ux_values[i][index])
        for (i, line) in enumerate(uy_lines):
            line.set_ydata(uy_values[i][index])
        for (i, line) in enumerate(vx_lines):
            line.set_ydata(vx_values[i][index])
        for (i, line) in enumerate(vy_lines):
            line.set_ydata(vy_values[i][index])
        
        plt.suptitle(f"{label}\nTime: {int(time[index])} ms")

    fig.subplots_adjust(top=0.85)

    return fig, update, len(time)



def animate_over_time(bf_file, files, sigmas):

    data = []

    for f_in in files:
        data.append(np.load(f_in, allow_pickle=True).item())
    
    mps_data = mps.MPS(bf_file)
    framerate = 0.2*mps_data.framerate
    
    x_lim = mps_data.frames.shape[0]*mps_data.info["um_per_pixel"]

    y_coord = 30
    blocksize = mps_data.frames.shape[0] // data[0]["all_values"]["displacement"].shape[1]
    fname = get_filename(bf_file, "org_image")
    plot_y_coord(mps_data.frames, y_coord, blocksize, fname)
    
    for key in ["displacement", "principal_strain"]:
        fname = get_filename(bf_file, key)
        fig, update, num_time_steps = _setup_2plots(data, sigmas, key, x_lim, y_coord, 0)
        _make_animation(fig, update, framerate, num_time_steps, fname)
        
        peak = np.argmax(mc.calc_norm_over_time(data[3]["all_values"][key]))
        _setup_2plots(data, sigmas, key, x_lim, y_coord, peak)
        plt.savefig(fname + ".png", bbox_inches='tight')
        plt.clf()
    
    shifts = [1, 0]

    for (key, shift) in zip(["deformation_tensor", "Green-Lagrange_strain_tensor"], shifts):
        fname = get_filename(bf_file, key)
        fig, update, num_time_steps = _setup_2x2plots(data, sigmas, key, x_lim, shift, y_coord, 0)
        _make_animation(fig, update, framerate, num_time_steps, fname)

        peak = np.argmax(mc.calc_norm_over_time(data[3]["all_values"][key]))
        _setup_2x2plots(data, sigmas, key, x_lim, shift, y_coord, peak)
        plt.savefig(fname + ".png", bbox_inches='tight')
        plt.clf()


bf_file = f"PointMM_4_ChannelBF_VC_Seq0000.nd2"
mc_files = []
sigmas = [0, 2, 4, 6, 8]

for s_txt in ["0", "2p0", "4p0", "6p0", "8p0"]:
    mc_file = f"PointMM_4_ChannelBF_VC_Seq0000/mpsmechanics/analyze_mechanics_block_size:3_matching_method:block_matching_max_block_movement:6_sigma:{s_txt}_type_filter:gaussian.npy"    
    mc_files.append(mc_file)

animate_over_time(bf_file, mc_files, sigmas)
