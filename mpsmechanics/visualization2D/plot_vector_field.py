"""

Module for plotting quiver and magnitude plots.

Åshild Telle / Simula Research Labratory / 2018-2019

"""


import sys
import os
import numpy as np
#import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as cl
import matplotlib as mpl
from math import ceil

from ..dothemaths import operations as op
from ..dothemaths import preprocessing as pp

def plot_vector_field(filename, title, xs, ys, U, V, arrow):
    """
    
    Gives a quiver plot of a given vector field.

    Arguments:
        filename - save as this file
        title - give title
        xs - x indices
        ys - y indices
        U, V - x and y components of vector field
        arrow - boolean value, plot with or without arrows

    """

    headwidth = 3 if arrow else 0

    plt.quiver(xs, ys, np.transpose(U), np.transpose(V), \
           headwidth=headwidth, minshaft=2.5)
    plt.title(title)
    plt.savefig(filename)
    plt.clf()


def plot_magnitude(vector_fields, norms, dimensions, titles, \
        path, idt):
    """
   
    Gives magnitude plots for given vector fields.

    Arguments:
        vector_fields - list of numpy arrays
        norms - list of colormap norms
        dimensions - scale each colour map accordingly
        titles - descriptions
        path - save here
        idt - using this attribute

    """

    scale = 10/dimensions[0]
    dimensions = (scale*dimensions[0], scale*dimensions[1])
    
    X, Y = vector_fields[0].shape[:2]
 
    xc = np.linspace(0, dimensions[0], X+1)
    yc = np.linspace(0, dimensions[1], Y+1)

    N = len(vector_fields)

    fig, ax = plt.subplots(N,1,figsize=dimensions)

    for n in range(N):
        plt.subplot(1, N, n+1)
        plt.pcolor(yc, xc, vector_fields[n], norm=norms[n], linewidth=0)
        plt.axis('off')
        plt.title(titles[n])

    plt.colorbar()
    filename = os.path.join(path, idt + "_magnitude.svg")
    plt.savefig(filename)
    #plt.show()
    plt.clf()



def plot_direction_and_magnitude(vector_fields, norms, labels, \
        dimensions, path, idt): 
    """

    Gives a combined quiver + colour plot for a set of given
    vector fields, combined in the same subplot.

    Arguments:
        vector_fields - list of a set of numpy array, each
            assumed to have the same dimension X x Y x 2
        norms - scale for colour plot
        labels - title given for each entry in the set
        dimensions - image size, for scaling to realistic
            x-y relation of the output plots
        path - save to this location
        idt - using this identity

    """

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

        data_d = op.normalize_values(data, over_time=False)
        data_m = op.calc_magnitude(data, over_time=False)
        
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

    filename = os.path.join(path, idt + "_direction_magnitude.png")
    plt.savefig(filename, dpi=1000)
    
    #plt.show()
    plt.clf()
