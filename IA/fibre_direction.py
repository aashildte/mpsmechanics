"""

Module for finding fibre direction from movement.

Åshild Telle / Simula Research Labratory / 2019

"""


import sys
import os
import numpy as np
import matplotlib.pyplot as plt

import io_funs as io
import preprocessing as pp
import mechanical_properties as mc
import heart_beat as hb
import least_sq_solver as lsq
import plot_vector_field as vs


def _preprocess_data(org, alpha, N_diff):
    """

    Preprocesses raw data + extracts data for timestep with largest displacement,
    to be used as input for the least squares solution. 

    Arguments:
        org - original displacement values, T x X x Y x 2 numpy array
        alpha - diffusion constant
        N_diff - number of times to do diffusion

    Return:
        normalized aligned averaged values, X x Y x 2 numpy array

    """

    T, X, Y = org.shape[:3]

    disp_t = pp.get_overall_movement(org)
    max_t = pp.get_max_ind(disp_t)

    data = pp.do_diffusion(np.asarray([org[max_t]]), alpha, N_diff)
    data = pp.normalize_values(data)
    data = pp.flip_values(data)

    # let direction be defined from timestep with largest displacement
        
    return data[0]


def _define_mesh_points(X, Y, dimensions):
    """

    Defines a mesh for the solution.

    Arguments:
        X - integer value, number of points in 1st dim
        Y - integer value, number of points in 2nd dim
        dimensions - dimensions of domain

    Returns:
        xs, ys - uniformly distributed values over
            [0, dimensions[0]] x [0, dimensions[1]]

    """

    dh_x = 0.5*dimensions[0]/X        # midpoints
    dh_y = 0.5*dimensions[1]/Y

    xs = np.asarray([dimensions[0]*i/X + dh_x for i in range(X)])
    ys = np.asarray([dimensions[1]*i/Y + dh_y for i in range(Y)])

    return xs, ys


def find_vector_field(data, M, N, basis_type, dimensions):
    """
        
    Finds a vector field representing the motion, using a least squares
    solver.

    Arguments:
        M, N - integer values, defines dimensions of a two-dimensional
            function space
        basis_type - defines basis functions, can be "trig" or "taylor"

    Returns:
        x and y components of vector field

    """
    
    X, Y = data.shape[1:3]
    xs, ys = _define_mesh_points(X, Y, dimensions)
    org_values = _preprocess_data(data, alpha = 0.75, N_diff = 5)

    disp_x, disp_y = org_values[:,:,0], org_values[:,:,1]
    l = lsq.Least_sq_solver(X, Y, xs, ys, disp_x, disp_y) 
    VX, VY = l.solve(M, N, basis_type)

    return VX, VY


if __name__ == "__main__":

    try:
        f_in, M, N, X, Y = sys.argv[1:6]
        M, N, X, Y = int(M), int(N), int(X), int(Y)
    except:
        print("Error: Give input file, M, N, length and height as arguments")
        exit(-1)

    x_len = 664E-6

    data, scale = io.read_disp_file(f_in, x_len)
    VX, VY = find_vector_field(data, M, N, "trig", [X, Y])


    print("VX, VY calculated – next step TBA")
