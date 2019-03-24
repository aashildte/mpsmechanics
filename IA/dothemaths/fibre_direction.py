"""

Module for finding fibre direction from movement.

Åshild Telle / Simula Research Labratory / 2019

"""


import sys
import os
import numpy as np
import matplotlib.pyplot as plt

import operations as op
import preprocessing as pp
import angular as an
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

    # get timestep with largest movement

    time_step = op.calc_max_ind(op.calc_norm_over_time(org))

    data_t = org[time_step]

    # treat/average values

    data_t = pp.do_diffusion(data_t, alpha, N_diff, over_time=False)
    data_t = op.normalize_values(data_t, over_time=False)
    data_t = an.flip_values(data_t, over_time=False)
 
    return data_t


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

    # unit tests

    M = N = X = Y = 3
    data = np.random.rand(3, 3, 3, 2)
    assert(find_vector_field(data, M, N, "trig", [X, Y]) is not None)

    print("All checks passed for fibre_direction.py")
