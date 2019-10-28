# -*- coding: utf-8 -*-

"""

Module for analyzing mechanical properties from motion vector images:
- Prevalence
- Displacement
- Principal strain

Åshild Telle / Simula Research Labratory / 2019

"""

import numpy as np


def calc_deformation_tensor(data, dx):
    """
    Computes the deformation tensor F from values in data

    Args:
        data - numpy array of dimensions T x X x Y x 2

    Returns:
        F, deformation tensor, of dimensions T x X x Y x 4

    """

    dudx = np.gradient(data[:, :, :, 0], axis=1)
    dudy = np.gradient(data[:, :, :, 0], axis=2)
    dvdx = np.gradient(data[:, :, :, 1], axis=1)
    dvdy = np.gradient(data[:, :, :, 1], axis=2)

    F = np.zeros(data.shape + (2,))

    F[:, :, :, 0, 0] = dudx
    F[:, :, :, 0, 1] = dudy
    F[:, :, :, 1, 0] = dvdx
    F[:, :, :, 1, 1] = dvdy

    F += np.eye(2)[None, None, None, :, :]

    return F


def calc_gl_strain_tensor(data, dx):
    """
    Computes the transpose along the third dimension of data; if data
    represents the deformation gradient tensor F (over time and 2 spacial
    dimensions) this corresponds to the Green-Lagrange strain tensor E.

    Args:
        data - numpy array of dimensions T x X x Y x 2 x 2

    Returns
        numpy array of dimensions T x X x Y x 2 x 2

    """

    F = calc_deformation_tensor(data, dx)
    C = np.matmul(F, F.transpose(0, 1, 2, 4, 3))
    E = 0.5*(C - np.eye(2)[None, None, None, :, :])

    return E

def calc_principal_strain(data, dx):
    """
    Computes the principal strain defined to be the largest eigenvector
    (eigenvector corresponding to largest eigenvalue, scaled) of the
    Cauchy-Green tensor, for each point (t, x, y).

    Args:
        data - displacement data, numpy array of dimension T x X x Y x 2

    Returns:
        principal strain - numpy array of dimension T x X x Y x 2

    """

    E = calc_gl_strain_tensor(data, dx)

    eigenvalues, eigenvectors = np.linalg.eig(E)

    eigen_filter = np.abs(eigenvalues[:, :, :, 0]) \
            > np.abs(eigenvalues[:, :, :, 1])
    eigen1 = eigenvalues[:, :, :, 0][:, :, :, None]\
            *eigenvectors[:, :, :, 0]
    eigen2 = eigenvalues[:, :, :, 1][:, :, :, None]\
            *eigenvectors[:, :, :, 1]

    return np.where(eigen_filter[:, :, :, None], eigen1, eigen2)
