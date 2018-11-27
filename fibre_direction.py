"""

Module for finding fibre direction from movement.

Åshild Telle / Simula Research Labratory / 2018

"""


import sys
import os
import numpy as np
import matplotlib.pyplot as plt

import least_sq_solver as lsq


class Fibre_direction:
    """

        Given experimental data on movement, this module does calculations
        on these and calculates a least squares solution for the movement
        direction, using a set of given basis functions.

    """
    def __init__(self, filename, rec_dimensions=(800, 170)):
        """
        
        Arguments:
            filename - first line time steps T and number of points in
                x and y direction given by X and Y, followed by T x X x Y
                lines displaying movement in x, y directions.
            rec_dimensions - tuple (x, y), if different from standard

        """

        org = self._read_disp_file(filename)
        T, X, Y = len(org), len(org[0]), len(org[0,0])

        self.xs, self.ys = self._define_mesh_pts(X, Y, rec_dimensions)
        self.T, self.X, self.Y, self.org = T, X, Y, org


    def _read_disp_file(self, filename):
        """

        Reads the input file, where the file is assumed to be a csv file on
        the form
            T, X, Y
            x0, y0
            x1, y1
            ...
        where we have T x X x Y values, giving the position for a given unit
        for time step t0, t1, ..., for  x coordinates x0, x1, ... and
        y coordinates y0, y1, ...

        x0, y0 gives relative motion of unit (0, 0) at time step 0; x1, y1
        gives relative motion of unit (1, 0) at time step 0, etc.

        Arguments:
            filename - csv file

        Returns:
            4-dimensional numpy array of rec_dimensions
                T x X x Y x 2

        """

        f = open(filename, 'r')
        T, X, Y = map(int, str.split(f.readline(), ","))

        org = np.zeros((T, X, Y, 2))

        for t in range(T):
            for i in range(X):
                for j in range(Y):
                    d = str.split(f.readline(), ",")[:2]
                    org[t, i, j] = np.array(d)

        f.close()

        return org

    def _get_max_timestep(self):
        """

        Finds time step with largest overall movement, defined by the
        combined 2-norm of relaitve movement of each unit.

        Returns:
            Integer value in interval [0, T]

        """

        org = self.org

        d_max = org[0]
        d_0 = org[0]
        max_norm = 0
        max_time_step = 0

        for t in range(len(org)):
            d = org[t]
            d_norm = np.linalg.norm(d-d_0)
            if(d_norm > max_norm):
                max_norm, max_time_step = d_norm, t

        return max_time_step

    def compute_gradient(self):
        """
        
            Computes the gradient from values saved in disp.

        """

        disp, T, X, Y = self.disp, self.T, self.X, self.Y

        grad_x = np.array(np.gradient(disp[:,:,:,0], axis=1))
        grad_y = np.array(np.gradient(disp[:,:,:,1], axis=2))
        grad = np.stack((grad_x, grad_y), axis=3)
        
        self.grad = grad

    def compute_deformation_tensor(self):
        """

            Computes the deformation tensor F from values saved in disp.

        """

        disp = self.disp

        dudx = np.array(np.gradient(disp[:,:,:,0], axis=1))
        dudy = np.array(np.gradient(disp[:,:,:,0], axis=2))
        dvdx = np.array(np.gradient(disp[:,:,:,1], axis=1))
        dvdy = np.array(np.gradient(disp[:,:,:,1], axis=2))
        
        F = np.stack((dudx, dudy, dvdx, dvdy), axis=3)

        self.F = F

    def compute_cauchy_green_tensor(self):
        """

            Computes the Cauchy-Green tensor C from values saved in disp.

        """
        T, X, Y = self.T, self.X, self.Y

        try:
            F = self.F
        except:
            self.compute_deformation_tensor()
        
        C = np.zeros_like(F)

        for t in range(T):
            for i in range(X):
                for j in range(Y):
                    C[t,i,j] = F[t,i,j].transpose()*F[t,i,j]

        self.C = C

    def _do_diffusion(self, disp, alpha, N_diff, t_start, t_stop):
        """

        Do diffusion/averaging of values using the molecule
              x
            x o x
              x
        weighting the middle point with value alpha and the surrounding
        points with value (1 - alpha).

        Boundary points are using molecule on the form

            x 0 x
              x

        Arguments:
            disp - numpy array, 2D or 3D
            alpha - weight, value between 0 and 1
            N_diff - number of times to run diffusion
            t_start - integer, from this value
            t_stop - integer, up to not including this value

        """
    
        X, Y = self.X, self.Y

        m = lambda a, i, j : alpha*a[i][j] + 0.25*(1 - alpha)* \
                (a[max(i-1,0)][j] + a[i][max(j-1,0)] +
                 a[min(i+1,X-1)][j] + a[i][min(j+1,Y-1)])

        for t in range(t_start, t_stop):
            disp1 = disp[t]                  # copy
            disp2 = np.zeros_like(disp1)

            # do diffusion

            for n in range(N_diff):
                for i in range(X):
                    for j in range(Y):
                        disp2[i][j] = m(disp1, i, j)

                disp1, disp2 = disp2, disp1       # swap pointers

            disp[t] = disp1                  # overwrite original

    def _normalize_values(self, values, t_start, t_stop):
        """
        
        Normalises each non-zero vector.

        Arguments:
            values - list to do operation on
            t_start - integer, from this value
            t_stop - integer, up to not including this value

        """
        
        X, Y = self.X, self.Y

        for t in range(t_start, t_stop):
            for i in range(X):
                for j in range(Y):
                    d_norm = np.linalg.norm(values[t, i,j])
                    if(d_norm > 1E-10):
                        values[t, i,j] = values[t, i,j]/d_norm


    def _flip_values(self, values, t_start, t_stop):
        """
        
        Define each vector to be in first or fourth quadrant

        Arguments:
            values - list to do operation on
            t_start - integer, from this value
            t_stop - integer, up to not including this value

        """
        X, Y = self.X, self.Y

        for t in range(t_start, t_stop):
            for i in range(X):
                for j in range(Y):
                    if(values[t,i,j,1] < 0):
                        values[t,i,j] = -values[t,i,j]

    def preprocess_data(self, alpha, N_diff, t='All', \
            diffuse_values=True, normalise_values=True, flip_values=True):
        """

        Transform raw data to prepared data, to be used as input for the
        least squares solution.

        Arguments:
            alpha - diffusion constant
            N_diff - number of times to do diffusion
            t - time steps of interest, can be integer, tuple or 'All'
            diffuse_values - boolean, do diffusion or not
            normalise_values - boolean, normalise values or not
            flip_values - boolean, flip values to first/fourth q or not

        """

        org, X, Y, T = self.org, self.X, self.Y, self.T

        t_start, t_stop = self._formate_time_values(t)

        disp = org[:]               # copy

        if(diffuse_values):
         self._do_diffusion(disp, alpha, N_diff, t_start, t_stop)

        if(normalise_values):
            self._normalize_values(disp, t_start, t_stop)
        
        if(flip_values):
            self._flip_values(disp, t_start, t_stop)

        # let direction be defined from displacement having
        # largest difference in absolute value, calculated from org. values
        
        disp_t = disp[self._get_max_timestep()]
        
        self.disp = disp
        self.disp_t = disp_t

    def _define_mesh_pts(self, X, Y, rec_dimensions):
        """

        Defines a mesh for the solution.

        Arguments:
            X - integer value, number of points in 1st dim
            Y - integer value, number of points in 2nd dim
            rec_dimensions - dimensions of domain

        Returns:
            xs, ys - uniformly distributed values of resp
                rec_dimensions[0], rec_dimensions[1]

        """

        dh_x = 0.5*rec_dimensions[0]/X        # midpoints
        dh_y = 0.5*rec_dimensions[1]/Y

        xs = np.asarray([rec_dimensions[0]*i/X + dh_x for i in range(X)])
        ys = np.asarray([rec_dimensions[1]*i/Y + dh_y for i in range(Y)])

        return xs, ys

    def find_vector_field(self, M, N, basis_type):
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
        
        X, Y, xs, ys, disp_t = \
            self.X, self.Y, self.xs, self.ys, self.disp_t

        disp_x, disp_y = disp_t[:,:,0], disp_t[:,:,1]

        l = lsq.Least_sq_solver(X, Y, xs, ys, disp_x, disp_y) 

        VX, VY = l.solve(M, N, basis_type)

        return VX, VY
    
    def _formate_time_values(self, t):
        """

        Gives user different choices for time step formate.

        Arguments:
            t: can be either integer value, list or tuple
                of 2 integer values or string 'All'
        
        Returns:
            t_start, t_stop - integers, time step range

        """

        if(type(t) is int):
            t_start, t_stop = t, t+1
        elif((type(t) is list) or (type(t) is tuple)):
            t_start, t_stop = t[0], t[1]
        elif(t is 'All'):
            t_start, t_stop = 0, self.T
        else:
            print("Unknown t value")
            exit(-1)

        return t_start, t_stop

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

    def plot_mechanical_properties(fd, t, path):
        """ plots gradient, deformation tensor F, cauchy-green tensor C """
    
        fd.compute_gradient()
        fd.compute_deformation_tensor()
        fd.compute_cauchy_green_tensor()

        # extract values to plot

        F_diag = np.stack([fd.F[:,:,:,0], fd.F[:,:,:,3]], axis=3)
        F_offdiag = np.stack([fd.F[:,:,:,1], fd.F[:,:,:,2]], axis=3)
        C_diag = np.stack([fd.C[:,:,:,0], fd.C[:,:,:,3]], axis=3)
        C_offdiag = np.stack([fd.C[:,:,:,1], fd.C[:,:,:,2]], axis=3)

        properties = [fd.org, fd.disp, fd.grad, F_diag, F_offdiag, C_diag, C_offdiag]
        labels = ("Original values", "Displacement", "Gradient", \
                "F diagonal values", "F off-diagonal values", \
                "C diagonal values", "C off-diagonal values")

        fd.plot_properties(properties, labels, path, t=t, arrow=True)

    def solve_linear_system(fd, path):
        """ solves linear system, plots solutions """
        
        # solve linear system, full rank

        M, N = 118, 25
        VX, VY = fd.find_vector_field(M, N, "trig")

        fd.plot_solution(path + "funspace.svg", \
                "Function space solution (full rank)", \
                VX, VY)

        # solve linear system, reduced rank

        M, N = 50, 12

        VX, VY = fd.find_vector_field(M, N, "trig")
    
        fd.plot_solution(path + "funspace2.svg", \
                "Function space solution (reduced rank)", \
                VX, VY)


    if(len(sys.argv) < 2):
        print("Error: Give input file as argument")
        exit(-1)

    filename = sys.argv[1]

    filename_prefix = filename.split(".")[0]
    print(filename_prefix)

    parent_path = "./Figures/"
    fig_path = "./Figures/" + filename_prefix + "/"

    for p in (parent_path, fig_path):
        if not (os.path.exists(p)):
            os.mkdir(p)
    
    fd = Fibre_direction(filename)
    t = fd._get_max_timestep()
    
    fd.preprocess_data(alpha = 0.75, N_diff = 5, t=t, \
            normalise_values=False, flip_values=False)

    plot_mechanical_properties(fd, t, fig_path)
    #solve_linear_system(fd, fig_path)
    

