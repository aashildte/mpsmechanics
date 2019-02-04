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

    X, Y, T = data.shape[:3]

    C = compute_cauchy_green_tensor(data)

    f = lambda x, i, j, \
            find_ps=lambda X : X[0][0]*X[1][1] if X[0][0] > X[0][1] \
                                    else X[0][1]*X[1][1] : \
            find_ps(np.linalg.eig(x))

    return pp.perform_operation(C, f, shape=(X, Y, T, 2))


def plot_properties(properties, labels, path, arrow=False):
        """
        
        Plots given data for given time step(s).

        Arguments:
            properties - list of numpy arrays, each needs four-dimensional
                of the same size, with the last dimension being 2 (normally
                each would be of dimension T x X x Y x 2).
            labels - for title and figure name
            path - where to save the figures
            t - time steps of interest, can be a single integer or a tuple
            arrow - boolean value, plot values with or without arrrow head

        """
    
        for (l, p) in zip(labels, properties):
            for t in range(T):
                filename = path + l + ("%03d" %t) + ".svg"
                x, y = p[t,:,:,0], p[t,:,:,1]
                fd.plot_solution(filename, (l + " at time step %3d" % t),
                    x, y, arrow=arrow)

def plot_solution(filename, title, U, V, arrow=False):
        """

        Gives a quiver plot.

        Arguments:
            filename - save as this file
            title - give title
            U, V - x and y components of vector field
            arrow - boolean value, plot with or without arrows

        """

        xs, ys = self.xs, self.ys

        headwidth = 3 if arrow else 0

        plt.subplot(211)
        plt.quiver(xs, ys, np.transpose(U), np.transpose(V), \
                headwidth=headwidth, minshaft=2.5)
        plt.title(title)
        plt.savefig(filename)
        plt.clf()



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
