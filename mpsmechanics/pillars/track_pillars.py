# -*- coding: utf-8 -*-
"""

Point tracking related functions

Åshild Telle / Simula Research Labratory / 2019

"""

import os
import numpy as np
import random
import matplotlib.pyplot as plt

import mpsmechanics as mc


def _define_pillars(p_values, N=200):
    """

    Defines circle to mark the pillar's circumference, based on
    information about the pillar's initial middle position.

    Args:
        p_values - list like structure of dimensions P x 3
        N - number of mesh points used

    Returns:
        p_values - mesh points on the circumsphere of the circle
            defined by x, y, z
    """

    assert(len(p_values) > 0), "p_values can't be an empty list"
    assert(N > 0), "N must be greater than 0"

    P = len(p_values)
    pillars = np.zeros((P, N, 2))
    angles = np.linspace(0, 2*np.pi, N)

    for i in range(P):
        '''
        ### mesh points on the circumsphere of the circle
        x, y, r = p_values[i]
        for j in range(N):
            pillars[i, j, 0] = x + r*np.cos(angles[j])
            pillars[i, j, 1] = y + r*np.sin(angles[j])
        '''

        ### CREATE A REDUCED RANDOM RADIUS
        x, y, r = p_values[i]
        reduced_radius = r - 1
        for j in range(N):
            random_radius = np.random.uniform(1.0, reduced_radius)  # put the N points inside of the detection circle
            pillars[i, j, 0] = x + random_radius * np.cos(angles[j])
            pillars[i, j, 1] = y + random_radius * np.sin(angles[j])

            #print('Coordinates: x =', x, ' , y = ', y, ' and radius = ', random_radius)

        '''
        ### REDUCED RADIUS
        x, y, r = p_values[i]
        reduced_radius = r - 2 #radius in µm
        for j in range(N):
            pillars[i, j, 0] = x + reduced_radius * np.cos(angles[j])
            pillars[i, j, 1] = y + reduced_radius * np.sin(angles[j])

            print('Coordinates: x =', x, ' , y = ', y, ' and radius = ', reduced_radius)
            
        '''
    return pillars


def _calculate_current_timestep(xs, ys, data, pillars):
    """

    Calculates values at given tracking points (defined by pillars)
    based on interpolation.

    Args:
        xs - x coordinates, dimension X
        ys - y coordinates, dimension Y
        data[t] - numpy array of dimensions X x Y x 2
        pillars - numpy array of dimensions P x N x 2

    Returns:
        numpy array of dimension P x N x 2, all points
            on the circumfence; absolute displacement
        numpy array of dimension N x 2, middle point (average
            of all points on the circumfence); relative displacement

    """
 
    fn_abs, fn_rel = mc.interpolate_values_2D(xs, ys, data)

    P, N, d = pillars.shape

    all_values = np.zeros((P, N, d))
    midpt_values = np.zeros((P, d))

    # all points, absolute displacement
    for p in range(P):
        for n in range(N):
            all_values[p, n] = fn_abs(pillars[p, n, 0], \
                    pillars[p, n, 1])

    # midpoints, relative displacement

    for p in range(P):
        mean = 0
        for n in range(N):
            mean += fn_rel(*pillars[p, n])
        mean /= N
        midpt_values[p] = mean

    return all_values, midpt_values


def _track_pillars_over_time(data, pillars_mpoints, dimensions):
    """

    Tracks position of mesh poinds defined for pillars over time, based
    on the displacement data.

    Args:
        data - displacement data, numpy array of size T x X x Y x 2
        mpoints - mesh points; center of each pillar
        dimensions - mesh size, 2-tuple

    Returns:
        numpy array of dimensions T x N x P (N default value);
            gives absolute displacement of circumfence points
        numpy array of dimensions T x P; gives relative displacement
            of middle point (i.e. the average)

    """

    # define pillars by their circumference

    pillars = _define_pillars(pillars_mpoints)
    
    # some general values
    T, X, Y = data.shape[:3]
    P, N = pillars.shape[:2]
    
    xs = np.linspace(0, dimensions[0], X)
    ys = np.linspace(0, dimensions[1], Y)

    # store data: for tracking circumfence; for middle points

    all_values = np.zeros((T, P, N, 2))
    midpt_values = np.zeros((T, P, 2))
    
    for t in range(T):
        all_values[t], midpt_values[t] = \
                _calculate_current_timestep(xs, ys, data[t], pillars)

    return all_values, midpt_values


def _scale_values(pillars_mpoints, data, scaling_factor, dimensions):
    """
    Args:
        scale_data - boolen value
        pillars_mpoints - numpy array of position and radii
        data - numpy array of displacement (T x X x Y x 2)

    Returns:
        pillars_mpoints - scaled, if applicable
        data - scaled, if applicable

    """
    print("Scale conversion: pixels (x, y, r) -> um (x, y, r)")
    
    for p in pillars_mpoints:
        print("(%d, %d, %d) -> (%d, %d, %d)" % (p[0], p[1], p[2], \
                                                int(scaling_factor * p[0]), int(scaling_factor * p[1]), int(scaling_factor * p[2])))

    data = scaling_factor * data # converts pixels to µm
    pillars_mpoints = scaling_factor * pillars_mpoints
    tmp_y = dimensions[0] * scaling_factor
    tmp_x = dimensions[1] * scaling_factor
    dimensions = np.array((tmp_y, tmp_x))

    return pillars_mpoints, data, dimensions


def _save_values(all_values, midpt_values, force_values, \
        calc_properties, pillars_mpoints, paths, data):
    """

    Args:
        all_values - circumference values, absolute displacement
        midpt_values - middle value, relative displacement
        force_values - middle value, force measurement
        calc_properties - which values to save
        pillars_mpoints - initial coordinates of midpoints
        paths_dir - dictionary of output folders

    """

    # find values at maximum displacement
        
    max_indices = mc.calc_beat_maxima_2D(data, \
           mc.calc_filter(data, 1E-10))

    # save values

    if (0 in calc_properties):
        mc.write_all_values(all_values, pillars_mpoints, \
                paths["num_all"])
    
    if(1 in calc_properties):
        if(len(max_indices)==0):
            print("No maxima found for this data set.")
        else:
            mc.write_max_values(midpt_values, max_indices, \
                    pillars_mpoints, paths["num_max"], "disp")

    if(2 in calc_properties):
        if(len(max_indices)==0):
            print("No maxima found for this data set.")
        else:
            mc.write_max_values(force_values, max_indices, \
                    pillars_mpoints, paths["num_max"], "force")


def _plot_values(all_values, midpt_values, force_values, pillars_mpoints,\
        plt_properties, paths):
    """

    Args:
        all_values - circumference values, absolute displacement
        midpt_values - middle value, relative displacement
        force_values - middle value, force measurement
        plt_properties - which values to save
        pillars_mpoints - initial coordinates of midpoints
        paths_dir - dictionary of output folders
    """
    if(0 in plt_properties):
        for t in range(len(all_values)):
            mc.plot_xy_coords(all_values[t], dimensions, \
                    t, paths["plt_all"])
    
    if(1 in plt_properties):
        mc.plot_over_time(midpt_values, pillars_mpoints, \
                paths["plt_max"], "disp")
    
    if(2 in plt_properties):
        mc.plot_over_time(force_values, pillars_mpoints, \
                paths["plt_max"], "force")


def track_pillars(f_disp, f_pts, calc_properties, plt_properties, scale_data):
    """

    Tracks points corresponding to "pillars" over time.

    Arguments:
        f_disp - filename for displacement data
        f_pts - filename for pillar positions data
        calc_properties - list of integers defining which properties to calculate
        plt_properties - list of integers defining which properties to plot
        scale_data - boolean value for scaling data to SI units or not

    """

    assert (".csv" in f_disp) or (".nd2" in f_disp), \
        "Displacement file must be a csv or nd2 file"
    assert ".csv" in f_pts, \
        "Pillar position file must be a csv file"
    assert all(isinstance(x, int) for x in calc_properties), \
        "calc_properties needs to be list of integer"
    assert all(isinstance(x, int) for x in plt_properties), \
        "plt_properties needs to be list of integer"
    assert type(scale_data) == bool, \
        "scale_data needs to be a boolean value"


    # force transformation ---- TODO cl arguments as well
    L = 50e-6  # in m
    R = 10e-6  # in m
    E = 2.63e6  # ????
    scale_data = True
    do_averaging = True

    area = L * R * np.pi * 1e6  # area in mm^2 half cylinder area

    # displacement data and positions of pillars
    data, scaling_factor, dimensions = mc.read_mt_file(f_disp)

    if do_averaging:
        data = mc.do_diffusion(data, alpha=0.75, N_diff=5, over_time=True)

    pillars_mpoints = mc.read_pt_file(f_pts, scaling_factor)

    if scale_data:
        pillars_mpoints, data, dimensions = _scale_values(
                               pillars_mpoints, data, scaling_factor, dimensions)
    else:
        print("Warning: Force measurements calculated without scaling. "
              + "Indicate scaling using -s or --scale to turn on"
              + "scaling of displacement.")
        dimensions = np.array(dimensions)

    # setup for saving things
    paths_dir, idt = mc.define_paths(f_disp, out_dir='track_pillars_MPS_0_7_random_radius_200_pts')

    print("Tracking pillars for data set: ", idt)
    # track values
    all_values, midpt_values = _track_pillars_over_time(data, pillars_mpoints, dimensions)

    midpt_values_meters = 1e-6 * midpt_values # converts data into meters
    force_values = mc.displacement_to_force_area(midpt_values_meters, E, L, R, area)

    _save_values(all_values, midpt_values, force_values, calc_properties,pillars_mpoints,
                paths_dir, data)

    _plot_values(all_values,midpt_values,force_values,pillars_mpoints,plt_properties,paths_dir)

    path_num = paths_dir["num_all"] + "; " + paths_dir["num_max"]

    print("Pillar tracking for " + idt + " finished:")
    print(" * Output saved in '" + path_num + "'")

    if len(plt_properties) > 0:
        path_plots = paths_dir["plt_all"] + "; " + paths_dir["plt_max"]
        print(" * Specified plots saved in '" + path_plots + "'")
