"""

Module for plotting quiver plots.

Åshild Telle / Simula Research Labratory / 2018

"""


import sys
import os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as cl
from math import ceil

import preprocess_data as pp
import read_data as io

def plot_vector_field(filename, title, xs, ys, U, V, arrow):
    """
    Gives a quiver plot.

    Arguments:
        filename - save as this file
        title - give title
        U, V - x and y components of vector field
        arrow - boolean value, plot with or without arrows

    """


    headwidth = 3 if arrow else 0

    plt.quiver(xs, ys, np.transpose(U), np.transpose(V), \
           headwidth=headwidth, minshaft=2.5)
    plt.title(title)
    plt.savefig(filename)
    plt.clf()

def plot_direction_and_magnitude(vector_fields, norms, labels, filters, dimensions, idt):

    path = "../Plots_dir_mag"
    io.make_dir_structure(path)

    N = len(vector_fields)
        
    k = 3

    X, Y = vector_fields[0].shape[:2]
 
    xc = np.linspace(0, dimensions[0], X+1)
    yc = np.linspace(0, dimensions[1], Y+1)

    # scale dimensions to something reasonable

    scale = 10/dimensions[0]
    dimensions = (scale*dimensions[0], scale*dimensions[1])

    plt.subplots(2*N,1,figsize=dimensions)

    for i in range(N):
        data = vector_fields[i]

        data_d = pp.normalize_values(np.asarray([data]))[0]
        data_m = pp.calculate_magnitude(np.asarray([data]))[0]
        
        data_d = filters[i](data_d)

        Ud, Vd = data_d[:,:,0], data_d[:,:,1]

        # get ever k value
        Ud = np.array([[Ud[i, j] for j in range(0, Y, k)] \
                for i in range(0, X, k)])
        Vd = np.array([[Vd[i, j] for j in range(0, Y, k)] \
                for i in range(0, X, k)])
        

        plt.subplot(1, 2*N, 1+2*i)
        plt.axis('off')
        plt.title(labels[i])
        plt.quiver(Ud, Vd, headwidth=3)

        plt.subplot(1, 2*N, 2*i+2)
        plt.axis('off')
        plt.title(" ")

        plt.pcolor(yc, xc, data_m, norm=norms[i], linewidth=0)


    #plt.savefig(path + "/" + idt + "_direction_magnitude.png")
    plt.savefig(path + "/" + idt + "_direction_magnitude.svg")
    #plt.show()
    plt.clf()

