# -*- coding: utf-8 -*-

"""

Åshild Telle / Simula Research Labratory / 2019

"""

import numpy as np


def calc_gradients(data, dx):
    """
    Computes gradients u_x, u_y, v_x, v_y from values in data

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

    gradients = np.zeros(data.shape + (2,))

    gradients[:, :, :, 0, 0] = dudx
    gradients[:, :, :, 0, 1] = dudy
    gradients[:, :, :, 1, 0] = dvdx
    gradients[:, :, :, 1, 1] = dvdy

    return gradients


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


def calc_gl_strain_tensor(def_tensor):
    """
    Computes the transpose along the third dimension of data; if data
    represents the deformation gradient tensor F (over time and 2 spatial
    dimensions) this corresponds to the Green-Lagrange strain tensor E.

    Args:
        numpy array of dimensions T x X x Y x 2 x 2

    Returns
        numpy array of dimensions T x X x Y x 2 x 2

    """

    def_tensor_transp = def_tensor.transpose(0, 1, 2, 4, 3)

    right_cg_tensor = np.matmul(def_tensor, def_tensor_transp)
    gl_strain_tensor = 0.5*(right_cg_tensor - np.eye(2)[None, None, None, :, :])

    return gl_strain_tensor


def calc_principal_strain(gl_strain_tensor):
    """
    Computes the principal strain defined to be the largest eigenvector
    (eigenvector corresponding to largest eigenvalue, scaled) of the
    Cauchy-Green tensor, for each point (t, x, y).

    Args:
        numpy array of dimensions T x X x Y x 2 x 2

    Returns:
        numpy array of dimension T x X x Y x 2

    """

    eigenvalues, eigenvectors = np.linalg.eig(gl_strain_tensor)

    eigen_filter = np.abs(eigenvalues[:, :, :, 0]) \
            > np.abs(eigenvalues[:, :, :, 1])
    eigen1 = eigenvalues[:, :, :, 0][:, :, :, None]\
            *eigenvectors[:, :, :, 0]
    eigen2 = eigenvalues[:, :, :, 1][:, :, :, None]\
            *eigenvectors[:, :, :, 1]

    return np.where(eigen_filter[:, :, :, None], eigen1, eigen2)


def calc_principal_strain_from_disp(displacement_data, dx):
    """
    Computes the principal strain defined to be the largest eigenvector
    (eigenvector corresponding to largest eigenvalue, scaled) of the
    Cauchy-Green tensor, for each point (t, x, y).

    Args:
        displacement_data - numpy array of dimensions T x X x Y x 2
        dx - float; spatial difference between two points/blocks

    Returns:
        numpy array of dimension T x X x Y x 2

    """

    def_tensor = calc_deformation_tensor(displacement_data, dx)
    gl_strain_tensor = calc_gl_strain_tensor(def_tensor)

    eigenvalues, eigenvectors = np.linalg.eig(gl_strain_tensor)

    eigen_filter = np.abs(eigenvalues[:, :, :, 0]) \
            > np.abs(eigenvalues[:, :, :, 1])
    eigen1 = eigenvalues[:, :, :, 0][:, :, :, None]\
            *eigenvectors[:, :, :, 0]
    eigen2 = eigenvalues[:, :, :, 1][:, :, :, None]\
            *eigenvectors[:, :, :, 1]

    return np.where(eigen_filter[:, :, :, None], eigen1, eigen2)
