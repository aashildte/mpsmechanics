"""

Module for plotting quiver plots.

Åshild Telle / Simula Research Labratory / 2018

"""


import sys
import os
import numpy as np
import matplotlib.pyplot as plt


def plot_solution(filename, title, xs, ys, U, V, arrow):
    """
    Gives a quiver plot.

    Arguments:
        filename - save as this file
        title - give title
        U, V - x and y components of vector field
        arrow - boolean value, plot with or without arrows

    """


    headwidth = 3 if arrow else 0

    plt.subplot(211)
    plt.quiver(xs, ys, np.transpose(U), np.transpose(V), \
           headwidth=headwidth, minshaft=2.5)
    plt.title(title)
    plt.savefig(filename)
    plt.clf()



def plot_properties(properties, labels, path, arrow):
    """
    Plots given data for given time step(s).

    Arguments:
        properties - list of numpy arrays, each needs four-dimensional
            of the same size, with the last dimension being 2 (normally
            each would be of dimension T x X x Y x 2).
       labels - for title and figure name
       path - where to save the figures
       arrow - boolean value, plot values with or without arrrow head

    """
    
    for (l, p) in zip(labels, properties):
        filename = path + l + ("%03d" %t) + ".svg"
        x, y = p[t,:,:,0], p[t,:,:,1]
        plot_solution(filename, (l + " at time step %3d" % t),
            x, y, arrow=arrow)

