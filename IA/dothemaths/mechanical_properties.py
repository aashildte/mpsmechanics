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

import dothemaths.operations as op

def calc_prevalence(movement, threshold):
    """
    Computes a true/false array for whether the displacement surpasses 
    a given threshold.

    This property only makes sense over time.

    Arguments:
        movement - T x X x Y x 2 numpy array
        threshold - cut-of value, in space/time units; should be scaled
            already - meaning that the movement can be compared on a
            unit scale
        over_time - boolean value; determines if T dimension
            should be included or not

    Returns:
        prevalence - (T - 1) x X x Y x 2 numpy boolean array

    """
 
    dxdt = np.diff(movement, axis=0)

    f_th = lambda x, i, j, th=threshold: (np.linalg.norm(x) > th)    

    return op.perform_operation(dxdt, f_th, over_time=True)


def _def_tensor_step(data):
    """

    Calculates the deformation gradient over a given X x Y data set.

    Arguments:
        data - numpy array of dimensions X x Y x 2

    Returns:
        deformation gradient for all points, numpy array of
        dimensions X x Y x 2 x 2

    """

    dudx = np.gradient(data[:,:,0], axis=0)
    dudy = np.gradient(data[:,:,0], axis=1)
    dvdx = np.gradient(data[:,:,1], axis=0)
    dvdy = np.gradient(data[:,:,1], axis=1)

    X, Y = data.shape[:2]
    F = np.zeros((data.shape + (2, )))

    for x in range(X):
        for y in range(Y):
            F[x, y] = np.array([[dudx[x, y] + 1, \
                                 dudy[x, y]], \
                                [dvdx[x, y], \
                                 dvdy[x, y] + 1]])

    return F


def calc_deformation_tensor(data, over_time):
    """
    Computes the deformation tensor F from values in data

    Arguments:
        data - numpy array of dimensions (T x) X x Y x 2
        over_time - boolean value; determines if T dimension
            should be included or not

    Returns:
        F, deformation tensor, of dimensions (T x) X x Y x 4

    """

    if(over_time):
        F = np.zeros((data.shape) + (2, ))
        T = data.shape[0]

        for t in range(T):
            F[t] = _def_tensor_step(data[t])

    else:
        F = _def_tensor_step(data)

    return F


def calc_cauchy_green_tensor(data, over_time):
    """
    Computes the transpose along the third dimension of data; if data
    represents the deformation gradient tensor F (over time and 2 spacial
    dimensions) this corresponds to the cauchy green tensor C.

    Arguments:
        data - numpy array of dimensions (T x) X x Y x 2 x 2
        over_time - boolean value; determines if T dimension
            should be included or not

    Returns
        C, numpy array of dimensions (T x) X x Y x 2 x 2

    """

    F = calc_deformation_tensor(data, over_time=over_time)
    f = lambda x, i, j : x.transpose()*x 

    return op.perform_operation(F, f, over_time=over_time)


def calc_principal_strain(data, over_time):
    """
    Computes the principal strain defined to be the largest eigenvector
    (eigenvector corresponding to largest eigenvalue, scaled) of the
    Cauchy-Green tensor, for each point (t, x, y).

    Arguments:
        data - displacement data, numpy array of dimension (T x) X x Y x 2
        over_time - boolean value; determines if T dimension
            should be included or not

    Returns:
        principal strain - numpy array of dimension (T x) X x Y x 2

    """

    C = calc_cauchy_green_tensor(data, over_time=over_time)

    f = lambda x, i, j, \
            find_ps=lambda S : S[0][0]*S[1][0] if S[0][0] > S[0][1] \
                                    else S[0][1]*S[1][1] : \
            find_ps(np.linalg.eig(x))

    P = op.perform_operation(C, f, over_time=over_time)

    return P


if __name__ == "__main__":

    data = np.random.rand(3, 3, 3, 2)
    threshold = 2*1E-6

    assert(calc_prevalence(data, threshold) is not None)
    assert(calc_deformation_tensor(data, over_time=True) is not None)    
    assert(calc_cauchy_green_tensor(data, over_time=True) is not None)
    assert(calc_principal_strain(data, over_time=True) is not None)

    print("All checks passed for mechanical_properties.py")
