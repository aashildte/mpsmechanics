"""

Module for analyzing mechanical properties from motion vector images:
- Prevalence
- Displacement
- Principal strain

Åshild Telle / Simula Research Labratory / 2019

"""


import sys
import os
import numpy as np
import matplotlib.pyplot as plt

import read_data as io
import preprocess_data as pp


def get_prevalence(movement, dt, dx, threshold):
    """
    Computes a true/false array for whether the displacement surpasses 
    a given threshold.

    Arguments:
        movement - T x X x Y x 2 numpy array
        dt - time difference (1/sampling rate)
        dx - space difference (width / X = height / Y)
        threshold - cut-of value, in space/time units

    Returns:
        prevalence - T x X x Y x 2 numpy boolean array

    """
 
    T, X, Y = movement.shape[:3]

    threshold = threshold*dx/dt         # scale

    f_th = lambda x, i, j, th=threshold: (np.linalg.norm(x) > th)    

    return pp.perform_operation(movement, f_th, shape=(T, X, Y))


def compute_deformation_tensor(data):
    """
    Computes the deformation tensor F from values in data

    TODO calculate using perform_operation

    Arguments:
        data - numpy array of dimensions T x X x Y x 2

    Returns:
        F, deformation tensor, of dimensions T x X x Y x 4

    """

    T, X, Y = data.shape[:3]

    dudx = np.array(np.gradient(data[:,:,:,0], axis=1))
    dudy = np.array(np.gradient(data[:,:,:,0], axis=2))
    dvdx = np.array(np.gradient(data[:,:,:,1], axis=1))
    dvdy = np.array(np.gradient(data[:,:,:,1], axis=2))

    F = np.zeros((T, X, Y, 2, 2))

    for t in range(T):
        for x in range(X):
            for y in range(Y):
                F[t, x, y] = \
                    np.array([[dudx[t, x, y] + 1, dudy[t, x, y]],
                              [dvdx[t, x, y], dvdy[t, x, y] + 1]])
    
    return F


def compute_cauchy_green_tensor(data):
    """
    Computes the transpose along the third dimension of data; if data
    represents the deformation gradient tensor F (over time and 2 spacial
    dimensions) this corresponds to the cauchy green tensor C.

    Arguments:
        data - numpy array of dimensions T x X x Y x 2 x 2

    Returns
        C, numpy array of dimensions T x X x Y x 2 x 2

    """

    F = compute_deformation_tensor(data)
    f = lambda x, i, j : x.transpose()*x 

    return pp.perform_operation(F, f)


def find_principal_vector(T, x, y):
    
    S = np.linalg.eig(T)

    if(S[0][0] > S[0][1]):
        return S[0][0]*S[1][0]
    else:
        return S[0][1]*S[1][1]




def compute_principal_strain(data):
    """
    Computes the principal strain defined to be the largest eigenvector
    (eigenvector corresponding to largest eigenvalue, scaled) of the
    Cauchy-Green tensor, for each point (t, x, y).

    Arguments:
        data - displacement data, numpy array of dimension T x X x Y x 2

    Returns:
        principal strain - numpy array of dimension T x X x Y x 2

    """

    T, X, Y = data.shape[:3]
    print("Dimensions: ", T, X, Y)
    
    C = compute_cauchy_green_tensor(data)

    #f = lambda x, i, j, \
    #        find_ps=lambda S : S[0][0]*S[1][0] if S[0][0] > S[0][1] \
    #                                else S[0][1]*S[1][1] : \
    #        find_ps(np.linalg.eig(x))

    P = pp.perform_operation(C, find_principal_vector, shape=(T, X, Y, 2))

    #for t in range(T):
    #    for x in range(X):
    #        for y in range(Y):
    #            print(P[t, x, y])

    return P

if __name__ == "__main__":

    try:
        f_in = sys.argv[1]
        idt = sys.argv[2]
    except:
        print("Error reading file names. Give displacement file name as " + \
             "first argument, identity as second.")
        exit(-1)
    
    data = io.read_disp_file(f_in)

    T, X, Y = data.shape[:3]

    assert(get_prevalence(data, 1, 1, T).shape == (T, X, Y, 2))
    print("Prevalence check passed")

    assert(compute_deformation_tensor(data).shape == (T, X, Y, 2, 2))
    print("F check passed")
    
    assert(compute_cauchy_green_tensor(data).shape == (T, X, Y, 2, 2))
    print("C check passed")

    assert(compute_principal_strain(data).shape == (T, X, Y, 2))
    print("Principal strain check passed")

    # TODO checks for plotting

    print("All checks passed")
