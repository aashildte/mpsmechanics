# -*- coding: utf-8 -*-

"""

Åshild Telle / Simula Research Labratory / 2019

"""

import numpy as np


def du7(u, axis, h):
    """ 
        Gives spacial derivative in one dimension using 7 pt stencil
        
        # TODO assert bcs

        Args:
            u
            axis - along which axis
            h = spacial step

        Returns:
            Array u' of spacial points of first derivative approximation
    """
  
    du = np.zeros_like(u)
    coefficients = (1./60, -3./20, 3./4, 0, -3./4, 3./20, -1./60)

    # use shift operator to calculate derivative
    for i in range(7):
        du += (1./h)*coefficients[i]*np.roll(u, i+3,axis=axis)

    return du


def calc_deformation_tensor(data, dx):
    """
    Computes the deformation tensor F from values in data

    Args:
        data - numpy array of dimensions T x X x Y x 2
        dx - float; spatial difference between two points/blocks

    Returns:
        numpy array of dimensions T x X x Y x 2 x 2

    """

    dudx = 1/dx*np.gradient(data[:, :, :, 0], axis=1)
    dudy = 1/dx*np.gradient(data[:, :, :, 0], axis=2)
    dvdx = 1/dx*np.gradient(data[:, :, :, 1], axis=1)
    dvdy = 1/dx*np.gradient(data[:, :, :, 1], axis=2)
    
    #dudx = du7(data[:,:,:,0], 1, dx)
    #dudy = du7(data[:,:,:,0], 2, dx)
    #dvdx = du7(data[:,:,:,1], 1, dx)
    #dvdy = du7(data[:,:,:,1], 2, dx)

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
