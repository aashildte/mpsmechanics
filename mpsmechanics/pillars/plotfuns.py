"""

TODO - split between a more general "pillar utils" file and iofuns
which can be used across the different scripts.

Åshild Telle / Simula Research Labratory / 2019

"""

import os
import numpy as np
import matplotlib.pyplot as plt


def plot_xy_coords(values, dimensions, t, path):
    """

    Scatter plot of given points, for a given time step. Units
    assumed to be micrometers.

    Args:
        values to be plotted - three-dimensional numpy array,
            last dimensions being 2 (x and y)
        dimensions - picture dimensions, to match output plot
            with actual picture dimensions
        t - timestep, used for identity
        path - save figures here

    """

    scale = 10/dimensions[0]
    dims_scaled = (dimensions[0]*scale, dimensions[1]*scale)

    x_vals = values[:, :, 0].flatten()
    y_vals = values[:, :, 1].flatten()
    
    t_id = "%04d" % t
    filename = os.path.join(path, "state_" + t_id + ".png")

    fg = plt.figure(t, figsize=dims_scaled)
    plt.xlim(0, dimensions[0])
    plt.ylim(0, dimensions[1])

    plt.xlabel("$\mu m$")
    plt.ylabel("$\mu m$")

    plt.scatter(x_vals, y_vals, s=0.01)
    plt.savefig(filename, dpi=1000)

    plt.close()


def plot_over_time(values, coords, path, prop):
    """

    Plots tracked values over time.

    Args:
        values - numpy array of dimension T x P x 2
        coords - coordinates; (x, y) for P pillars
        path - save figures here

    """
    
    T, P = values.shape[:2]
    fps = 100
    Tmax = T/fps
    ts = np.linspace(0, Tmax, T)

    if(prop == "disp"):
        titles = ["Displacement over time (x values)", \
              "Displacement over time (y values)", \
              "Displacement over time (l2 norm)"]
    else:
        titles = ["Force measurement over time (x values)", \
              "Force measurement over time (y values)", \
              "Force measurement over time (l2 norm)"]

    filenames = [os.path.join(path, prop + "_all_time_steps_x.png"), \
            os.path.join(path, prop + "_all_time_steps_y.png"), \
            os.path.join(path, prop + "_all_time_steps_norm.png")]

    x_values = values[:,:,0]
    y_values = values[:,:,1]

    overall_values = np.zeros((T, P))

    for t in range(T):
        for p in range(P):
            overall_values[t,p] = np.linalg.norm(values[t, p])
    
    d_values = [x_values, y_values, overall_values]

    for i in range(3):
        for p in range(P):
            plt.plot(ts, d_values[i][:,p])

        plt.title(titles[i])
        plt.legend([("(" + str(int(x[0])) + ", " + str(int(x[1])) + ")") \
                for x in coords])
        plt.xlabel("Time ($s$)")

        if(prop == "disp"):
            plt.ylabel("Displacement ($\mu m$)")
        elif(prop == "force"):
            plt.ylabel("Force ($N$)")
        plt.savefig(filenames[i], dpi=1000) 
        plt.close()

