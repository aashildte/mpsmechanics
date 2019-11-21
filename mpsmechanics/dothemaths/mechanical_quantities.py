# -*- coding: utf-8 -*-

"""

Åshild Telle / Simula Research Labratory / 2019

"""

from scipy.ndimage import gaussian_filter
from medpy.filter.smoothing import anisotropic_diffusion
import numpy as np


def calc_gradients(data, dx):
    dudx = 1/dx*np.gradient(data[:, :, :, 0], axis=1)
    dudy = 1/dx*np.gradient(data[:, :, :, 0], axis=2)
    dvdx = 1/dx*np.gradient(data[:, :, :, 1], axis=1)
    dvdy = 1/dx*np.gradient(data[:, :, :, 1], axis=2)
    
    gradients = [dudx, dudy, dvdx, dvdy]
    G = np.zeros(data.shape + (2,))

    G[:, :, :, 0, 0] = gradients[0]
    G[:, :, :, 0, 1] = gradients[1]
    G[:, :, :, 1, 0] = gradients[2]
    G[:, :, :, 1, 1] = gradients[3]

    return G


def calc_deformation_tensor(data, dx):
    """
    Computes the deformation tensor F from values in data

    Args:
        data - numpy array of dimensions T x X x Y x 2
        dx - float; spatial difference between two points/blocks

    Returns:
        numpy array of dimensions T x X x Y x 2 x 2

    """

    gradients = calc_gradients(data, dx)
    return gradients + np.eye(2)[None, None, None, :, :]


def calc_gl_strain_tensor(data, dx):
    """
    Computes the transpose along the third dimension of data; if data
    represents the deformation gradient tensor F (over time and 2 spatial
    dimensions) this corresponds to the Green-Lagrange strain tensor E.

    Args:
        data - numpy array of dimensions T x X x Y x 2 x 2
        dx - float; spatial difference between two points/blocks

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
        dx - float; spatial difference between two points/blocks

    Returns:
        numpy array of dimension T x X x Y x 2

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
