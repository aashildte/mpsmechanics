# -*- coding: utf-8 -*-

"""

Module for analyzing mechanical properties from motion vector images:
- Prevalence
- Displacement
- Principal strain

Åshild Telle / Simula Research Labratory / 2019

"""

import numpy as np

from . import operations as op

def calc_prevalence(data, threshold):
    """

    Computes a true/false array for whether the displacement surpasses
    a given threshold.

    This property only makes sense over time.

    Args:
        data - T x X x Y x 2 numpy array
        threshold - cut-of value, in space/time units; should be scaled
            already - meaning that the movement can be compared on a
            unit scale
        over_time - boolean value; determines if T dimension
            should be included or not

    Returns:
        prevalence - T x X x Y x 2 numpy boolean array

    """

    dxdt = np.gradient(data, axis=0)

    f_th = lambda x, i, j, th=threshold: (np.linalg.norm(x) > th)

    return op.perform_operation(dxdt, f_th, over_time=True)


def calc_deformation_tensor(data):
    """
    Computes the deformation tensor F from values in data

    Args:
        data - numpy array of dimensions T x X x Y x 2

    Returns:
        F, deformation tensor, of dimensions T x X x Y x 4

    """    

    dudx = np.gradient(data[:, :, :, 0], axis=0)
    dudy = np.gradient(data[:, :, :, 0], axis=1)
    dvdx = np.gradient(data[:, :, :, 1], axis=0)
    dvdy = np.gradient(data[:, :, :, 1], axis=1)

    F = np.swapaxes(np.swapaxes(np.swapaxes(np.swapaxes(\
            np.array(((dudx, dudy), \
                      (dvdx, dvdy))), \
                      0, -2), 1, -1), 0, 2), 1, 2) \
             + np.eye(2)[None, None, None, :, :]

    return F


def calc_cauchy_green_deformation_tensor(data):
    """
    Computes the transpose along the third dimension of data; if data
    represents the deformation gradient tensor F (over time and 2 spacial
    dimensions) this corresponds to the cauchy green tensor C.

    Args:
        data - numpy array of dimensions T x X x Y x 2 x 2

    Returns
        C, numpy array of dimensions T x X x Y x 2 x 2

    """

    F = calc_deformation_tensor(data)

    return 0.5*(np.matmul(F, F.transpose(0, 1, 2, 4, 3)) -  np.eye(2)[None, None, None, :, :])


def calc_principal_strain(data):
    """
    Computes the principal strain defined to be the largest eigenvector
    (eigenvector corresponding to largest eigenvalue, scaled) of the
    Cauchy-Green tensor, for each point (t, x, y).

    Args:
        data - displacement data, numpy array of dimension T x X x Y x 2

    Returns:
        principal strain - numpy array of dimension T x X x Y x 2

    """
    
    C = calc_cauchy_green_deformation_tensor(data)
    
    T, X, Y, _, _ = C.shape

    eigenvalues, eigenvectors = np.linalg.eig(C)

    eigen_filter = np.abs(eigenvalues[:,:,:,0]) > np.abs(eigenvalues[:,:,:,1])

    # TODO there must be a numpy function for this ..

    pr_strain = np.zeros((T, X, Y, 2))

    for t in range(T):
        for x in range(X):
            for y in range(Y):
                if(eigen_filter[t, x, y]):
                    pr_strain[t, x, y] = eigenvalues[t, x, y, 0]*\
                            eigenvectors[t, x, y, 0]
                else:
                    pr_strain[t, x, y] = eigenvalues[t, x, y, 1]*\
                            eigenvectors[t, x, y, 1] 
    
    return pr_strain
