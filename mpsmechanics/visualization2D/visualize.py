
"""

Åshild Telle / Simula Research Labratory / 2019

"""


import os
import sys
import numpy as np
import matplotlib.colors as cl
import matplotlib.pyplot as plt

import mpsmechanics as mc


def read_values(f_in, dimensions):
    """

    Get values of interest.

    Arguments:
        Filename - csv or nd2 file, ref. README
        dimensions - dimensions of each frame in um

    Returns:
        displacement - X x Y x 2 numpy array
        principal strain - X x Y x 2 numpy array
        e_alpha - direction of most detected movement
        e_beta - perpendicular direction vector

    """

    xlen = 664.30*1E-6

    disp_data, scale = io.read_file(f_in, xlen)
    time_step = op.calc_max_ind(op.calc_norm_over_time(disp_data))

    alpha, N_d = 0.75, 5
    e_alpha, e_beta = an.calc_direction_vectors(disp_data)

    disp_data_t = pp.do_diffusion(disp_data[time_step], alpha, N_d, \
            over_time=False)
    strain_data_t = mc.calc_principal_strain(disp_data_t, \
            over_time=False)
    
    return scale*disp_data_t, strain_data_t, e_alpha, e_beta


def plot_angle_distribution(data, e_ref, p_value, p_label, path, \
        title, idt):
    """
    
    Histogram plot, angular distribution for values with magnutide over
    given threshold

    TODO maybe separate functions for calculations and plotting?
    - if needed other places too.

    Arguments:
        data - X x Y x 2 numpy array
        e_ref - direction vector, "0 angle"
        p_value - threshold
        p_label - corresponding label
        path - save here
        title - description
        idt - attribute for plots

    """

    X, Y = data.shape[:2]

    values = []

    for x in range(X):
        for y in range(Y):
            norm = np.linalg.norm(data[x, y])
            if(norm > p_value and norm > 1E-14):
                ip = np.dot(data[x, y], e_ref)/norm
                ip = max(min(ip, 1), -1)      # eliminate small overflow
                angle = np.arccos(ip)
                values.append(angle)

    num_bins=100
    plt.hist(values, num_bins, density=True, alpha=0.5)
    plt.title(title)
    plt.xlim((0, np.pi))
    plt.savefig(os.path.join(path, idt + "_angle_distribution_" + \
            p_label + ".png"))
    plt.clf()    


def plot_thresholds(values, per_values, fractions, path, title, idt):
    """
    Plot value distribution (based on magnitude) and threshold; for
    visual check / description.

    Arguments:
        values - data, X x Y numpy array
        per_value - thersholds; calculated percentage values
        fractions  corresponding fractions; for labeling
        path - save here
        title - description
        idt - file identity

    """

    for (p, l) in zip(per_values, fractions):
        label = "%.1f" % l
        plt.axvline(x=p, label=label)

    num_bins=100
    plt.hist(values, num_bins, alpha=0.5, density=True)
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(path,idt + "_distribution.png"))
    plt.clf()


def plot_original_values(x_part, y_part, lognorm, path, idt, dimensions):
    """

    Plots x, y components; original magnitude.

    Arguments:
        x_part - x projection
        y_part - y projection
        lognorm - boolean value; plot on logscale or not
        path - save here
        idt - using this attribute
        dimensions - picture dimensions, for scaling of plots
    """

    max_val = max(np.max(x_part), np.max(y_part))
    
    if(lognorm):
        norm = cl.LogNorm(0.1, max_val + 0.1)
    else:
        norm = cl.Normalize(0.0, max_val)
        
    titles = ["X component", "Y component"]

    pl.plot_magnitude([x_part, y_part], 2*[norm], dimensions, \
            titles, path, idt + "_original_values")

def calc_percentile_values(org, z_dir, per_value):
    
    X, Y = org.shape[:2]

    new = np.zeros((X, Y))

    for x in range(X):
        for y in range(Y):
            if(np.linalg.norm(org[x, y]) > per_value):
                new[x, y] = np.linalg.norm(z_dir[x, y])/\
                        (np.linalg.norm(org[x, y]))
    return new
    

def plot_percentile_values(org, x_frac, y_frac, p_value, p_label, \
        lognorm, dimensions, title, path, idt):
    """

    Finding -> plotting colour maps for given values; x and y
    directions combined.

    Arguments:
        original data
        x_frac - x projection
        y_frac - y projection
        p_value - threshold for movement
        p_label - corresponding label (fraction)
        lognorm - boolean; use log scale or not
        dimensions - of original picture, used to scale plots
        title - description
        path - save here
        idt - using this attribute

    """

    x_part, y_part = [calc_percentile_values(org, z, p_value) \
                for z in [x_frac, y_frac]]

    max_val = max(np.max(x_part), np.max(y_part))

    if(lognorm):
        norm = cl.LogNorm(0.1, max_val + 0.1)
    else:
        norm = cl.Normalize(0.0, max_val)

    titles = [title + " X fraction", title + " Y fraction"]

    pl.plot_magnitude([x_part, y_part], 2*[norm], dimensions, \
            titles, path, idt + "_" + p_label)


def plot_values(values, e_alpha, e_beta, path, idt, dimensions, \
        title, lognorm):
    """

    Plots values of interest for given data set; this includes
        * angle distribution: histogram
        * 

    Arguments:
        values - values for given property; X x Y x 2 numpy array
        e_alpha - vector aligned with most detected movement
        e_beta - perpendicular vector
        path - save here
        idt - use as attribute when saving files
        dimensions - dimensions of recorded image, to scale plots
        title - description for plotting
        lognorm - boolean value; plot on a log scale or not

    """

    X, Y = values.shape[:2]

    # calculate relevant data
    values_m = op.calc_magnitude(values, over_time=False)

    x_values, y_values = [op.calc_magnitude(\
            an.calc_projection_vectors(values, e, over_time=False), \
            over_time=False) \
            for e in [e_alpha, e_beta]]

    # find thresholds
    N = 5
    fractions = np.linspace(0, .4, N)
    max_v = np.max(values_m)
    min_v = np.min(values_m)

    per_values = np.array([min_v + x*(max_v - min_v) \
            for x in fractions])
    
    plot_thresholds(values_m.reshape(X*Y), per_values, fractions, \
            path, title, idt)
    plot_original_values(x_values, y_values, lognorm, path, idt, \
            dimensions)

    for i in range(N):
        p_label = str(int(100*fractions[i]))

        # angle distribution, based on thersholds from above
        plot_angle_distribution(values, -e_beta, per_values[i], \
                p_label, path, title, idt)

        # colour maps
        plot_percentile_values(values, x_values, y_values, \
                per_values[i], p_label, lognorm, dimensions, \
                title, path, idt)
