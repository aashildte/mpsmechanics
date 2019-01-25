"""

Module for analyzing mechanical properties from motion vector images:
- Displacement
- Velocity
- Principal strain

Åshild Telle / Simula Research Labratory / 2019

"""


import sys
import os
import numpy as np
import matplotlib.pyplot as plt

import preprocess_data as pp


def get_prevalence(movement, dt, dx, threshold):
    """
    Arguments:
        movement - T x X x Y x 2 numpy array

    Returns:
        prevalence - T x X x Y x 2 numpy array

    """

    threshold = threshold*dx/dt            # in same units

    f_th = lambda x, i, j, threshold=threshold: (np.linalg.norm(x) > threshold)    

    return pp.perform_operation(movement, f_th)



def compute_velocity(data, dt):
    T, X, Y = pp.get_dimensions(data)

    v = np.zeros((T-1, X, Y, 2))

    for t in range(T-1):
        for x in range(X):
            for y in range(Y):
                v[t, x, y] = (data[t+1, x, y] - data[t, x, y])/dt

    return v


def compute_gradient(data):
    """
        Computes the gradient from values in data

    """
    T, X, Y = pp.get_dimensions(data)

    grad_x = np.array(np.gradient(data[:,:,:,0], axis=1))
    grad_y = np.array(np.gradient(data[:,:,:,1], axis=2))
    grad = np.stack((grad_x, grad_y), axis=3)
    
    return grad

def compute_deformation_tensor(disp):
    """
        Computes the deformation tensor F from values in data
    """


    dudx = np.array(np.gradient(disp[:,:,:,0], axis=1))
    dudy = np.array(np.gradient(disp[:,:,:,0], axis=2))
    dvdx = np.array(np.gradient(disp[:,:,:,1], axis=1))
    dvdy = np.array(np.gradient(disp[:,:,:,1], axis=2))
        
    F = np.stack((dudx, dudy, dvdx, dvdy), axis=3)

    return F

def compute_cauchy_green_tensor(F):
    """
        Computes the Cauchy-Green tensor C from F.
    """

    T, X, Y = len(F), len(F[0]), len(F[0,0])
 
    C = np.zeros_like(F)

    for t in range(T):
        for i in range(X):
            for j in range(Y):
                C[t,i,j] = F[t,i,j].transpose()*F[t,i,j]

    self.C = C


def plot_properties(self, properties, labels, path, t='All', arrow=False):
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
    
        t_start, t_stop = self._formate_time_values(t)

        for (l, p) in zip(labels, properties):
            for t in range(t_start, t_stop):
                filename = path + l + ("%03d" %t) + ".svg"
                x, y = p[t,:,:,0], p[t,:,:,1]
                fd.plot_solution(filename, (l + " at time step %3d" % t),
                    x, y, arrow=arrow)

def plot_solution(self, filename, title, U, V, arrow=False):
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
    pass
