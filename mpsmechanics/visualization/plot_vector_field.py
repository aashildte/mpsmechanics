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


def plot_vector_field(data, dimensions, filename):
    """
    
    Gives a quiver plot of a given vector field.

    Arguments:
        vector_field - X x Y numpy array
        dimensions - scale each colour map accordingly
        filename - for saving

    """

    X, Y, _ = data.shape
    
    # downscale data

    x_take, y_take = [[i for i in range(0, d, 2)] for d in (X, Y)]

    xs = np.linspace(0, dimensions[0], X)[x_take]
    ys = np.linspace(0, dimensions[1], Y)[y_take]

    U = data[:,:,0]
    U = np.take(U, x_take, axis=0)
    U = np.take(U, y_take, axis=1)
    V = data[:,:,1]
    V = np.take(V, x_take, axis=0)
    V = np.take(V, y_take, axis=1)

    scale = 10/dimensions[0]
    figsize = (scale*dimensions[0], scale*dimensions[1])
    plt.figure(figsize=figsize)

    plt.quiver(xs, ys, np.transpose(U), np.transpose(V))
    plt.savefig(filename)
    plt.close()


def plot_magnitude(data, dimensions, filename, norm):
    """
   
    Gives magnitude plots for given vector field.

    Arguments:
        vector_field - X x Y x 2 numpy array
        dimensions - scale each colour map accordingly
        filename - for saving
        norms - colormap norm

    """

    data_normed = op.calc_magnitude(data, False)

    if(norm is None):
        d_min = np.min(data_normed)
        d_max = np.max(data_normed)
        norm = mpl.colors.Normalize(vmin=d_min,vmax=d_max)

    scale = 10/dimensions[0]
    figsize = (scale*dimensions[0], scale*dimensions[1])
    plt.figure(figsize=figsize)

    X, Y = data.shape[:2]

    xc = np.linspace(0, dimensions[0], X+1)
    yc = np.linspace(0, dimensions[1], Y+1)
    
    plt.pcolor(xc, yc, np.transpose(data_normed), norm=norm, linewidth=0)
    plt.colorbar()
    plt.savefig(filename)

    plt.close()


def plot_direction(data, dimensions, filename):
    """

    Gives normalized quiver plots for given vector field.

    Arguments:
        vector_field - X x Y numpy array
        dimensions - scale each colour map accordingly
        filename - for saving

    """

    data_norm = op.normalize_values(data, over_time=False)
    plot_vector_field(data_norm, dimensions, filename)

