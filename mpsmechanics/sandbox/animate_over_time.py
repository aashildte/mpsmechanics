
import os
import sys
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

import mpsmechanics as mc
import mps
from mpl_toolkits.axes_grid1 import make_axes_locatable


def get_filename(input_file, quantity):
    folder = os.path.join(input_file[:-4], "animation_over_time")
    os.makedirs(folder, exist_ok=True)

    fname = os.path.join(folder, quantity)

    return fname


def values_BF(f_in):

    fname = get_filename(f_in, "displacement")
    data = mc.read_prev_layer(f_in, \
                "analyze_mechanics", mc.analyze_mechanics)
    
    time = data["time"]
    disp = data["over_time_avg"]["displacement"]

    return time, disp, r"Displacement ($\mu m$)", fname


def values_MPS(f_in):

    data = np.load(os.path.join(f_in[:-4], "data.npy"), allow_pickle=True).item()
    values = data["unchopped_data"]["trace"]
    time = data["unchopped_data"]["time"] 
    ylabel = "Fluorescence"

    if "Cyan" in f_in:
        fname = get_filename(f_in, "channel_cyan")
    else:
        fname = get_filename(f_in, "channel_red")

    return time, values, ylabel, fname


def animate_over_time(f_in):
    if "BF" in f_in:
        time, values, ylabel, fname = values_BF(f_in)
    elif "Cyan" in f_in or "Red" in f_in:
        time, values, ylabel, fname = values_MPS(f_in)
    else:
        return

    mps_data = mps.MPS(f_in)
    framerate = mps_data.framerate


    fig, axis = plt.subplots(figsize=(7, 2))
    plt.xlim(0, time[-1])
    plt.ylim(-0.1*max(values), 1.2*max(values))
    plt.xlabel("Time (ms)")
    plt.ylabel(ylabel)
    line, = axis.plot(time[0], values[0])
    
    def update(index):
        line.set_xdata(time[:index])
        line.set_ydata(values[:index])
    
    fig.tight_layout()
    
    writer = animation.writers["ffmpeg"](fps=framerate)
    anim = animation.FuncAnimation(fig, update, len(time))

    fname = os.path.splitext(fname)[0]
    anim.save("{}.{}".format(fname, "mp4"), dpi=300, writer=writer)

    plt.close()

    fig, axis = plt.subplots(figsize=(7, 2))
    plt.xlim(0, time[-1])
    plt.ylim(-0.1*max(values), 1.2*max(values))
    plt.xlabel("Time (ms)")
    plt.ylabel(ylabel)
    axis.plot(time, values)
    plt.tight_layout()

    plt.savefig(fname + ".png", dpi=600)

    plt.close()


f_in = sys.argv[1]

animate_over_time(f_in)
