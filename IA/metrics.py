"""

Given data on displacement, this script finds average and overall
values for a number of features: Average of each of
    - beat rate
    - displacement
    - x motion
    - prevalence
    - principal strain

Figures for alignment as well as all the characteristic values are
plotted and saved in "Plots"; each is saved both as a png and as a
svg file.

Åshild Telle / Simula Research Labratory / 2019

"""

import sys
import numpy as np

import io_funs as io
import preprocessing as pp
import heart_beat as hb
import mechanical_properties as mc
 

def get_xfraction(disp_data, disp_data_t, idt, dimensions):
    """
 
    Calculates x fraction of movement over all time steps.

    Arguments:
        disp_data - displacement, numpy array of dimensions
            T x X x Y x 2
        disp_data_t - displacement over time (L2 norm),
            numpy array of dimension T
        idt - identity for relevant plots
        dimensions - dimensions of picture for recording
    
    Returns:
        x motion over time, normalized wrt disp_data_t

    """

    e_alpha, e_beta = pp.find_direction_vectors(disp_data, idt, \
        dimensions)

    disp_data_x = pp.get_projection_vectors(disp_data, e_alpha)
    disp_data_x_t = pp.get_overall_movement(disp_data_x)
  
    T = len(disp_data_x_t)
 
    disp_data_xfraction = np.zeros(T)

    for t in range(T):
        if(disp_data_t[t] > 1E-10):
            disp_data_xfraction[t] = disp_data_x_t[t]/disp_data_t[t]
    
    return disp_data_xfraction



def get_prevalence(disp_data, disp_data_t, threshold):
    """
 
    Calculates prevalence over all time steps.

    Arguments:
        disp_data - displacement, numpy array of dimensions
            T x X x Y x 2
        disp_data_t - displacement over time (l2 norm), numpy
            array of dimension T
        threshold - should be scaled to unit scales 

    Returns:
        prevalence over time, normalized with respect to X and Y;
        numpy array of dimensions T-1 x X x Y x 2

    """

    T, X, Y = disp_data.shape[:3]

    # some parameters
    #dx = 2044./664*10E-6          # approximately

    # scale threshold:

    scale = 1./(X*Y)              # no of points
    
    prev_xy = mc.get_prevalence(disp_data, threshold)

    prevalence = np.zeros(T-1)

    for t in range(T-1):
        for x in range(X):
            for y in range(Y):
                if(prev_xy[t,x,y]==1):
                    prevalence[t] = prevalence[t] + 1

    return scale*prevalence


def get_numbers_of_interest(disp_data, scale, idt, dt, dx, \
        dimensions):
    """

    Arguments:
        disp_data - displacement data 
        idt - data set idt
        dt - temporal difference
        dx - spacial difference
        dimensions - tuple (x, y), picture dimensions

    Returns:
        List of values:
          average beat rate
          average displacement
          average displacement in x direction
          average prevalence

    where each value is taken over peak values only.

    """
    
    # beat rate - find all maximum points

    T, X, Y = disp_data.shape[:3]
    T_max = dt*T

    maxima = hb.get_beat_maxima(disp_data, scale, idt, T_max)

    beat = np.array([(maxima[k] - maxima[k-1]) \
                        for k in range(1, len(maxima))])

    # get displacement, displacement x, prevalance, principal strain

    disp_data_t = pp.get_overall_movement(disp_data)

    disp_data_x_fraction = get_xfraction(disp_data, disp_data_t, \
        idt, dimensions)
    
    # threshold given by 2 um/s; emperically, from paper
    # we scale to get on same units as displacement data,
    # and to get dt on a unit scale in prevalence test

    threshold = 2*10E-6*dt/scale

    prevalence = get_prevalence(disp_data, disp_data_t, threshold)

    # TODO check scaling

    pr_strain = pp.get_overall_movement(\
                     mc.compute_principal_strain(scale*disp_data))

    # ... for each beat ...

    values = [disp_data_t, disp_data_x_fraction, prevalence, \
        pr_strain]
    beat_values = np.array([[metric[m] for m in maxima] \
                   for metric in values])

    # plot some values
    suffixes = ["Displacement", "X motion", "Prevalence", \
                    "Principal strain"]

    yscales = [None, (-0.1, 1.1), (-0.1, 1.1), None]

    scaling = [scale, 1, 1, 1]   # TODO check scaling of ps

    for (sc, val, s, yscale) in zip(scaling, values, suffixes, yscales):
        hb.plot_maxima(sc*val, maxima, idt, s, T_max, yscale)

    # average

    if(len(maxima)==0):
        print("Empty sequence – no maxima found")
        r_values = []
    elif(len(maxima)==1):
        print("One single beat found")
        r_values = [np.mean(v) for (sc, v) in zip(scaling, beat_values)]
    else:
        r_values = [np.mean(beat)] + \
                [np.mean(v) for (sc, v) in zip(scaling, beat_values)]

    return r_values


# TODO implement unit tests


