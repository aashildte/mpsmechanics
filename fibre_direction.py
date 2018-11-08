"""

Module for finding fibre direction from movement.

Åshild Telle / Simula Research Labratory / 2018

"""


import sys
import numpy as np
import matplotlib.pyplot as plt

import least_sq_solver as lsq


class Fibre_direction:
    """

        Given experimental data on movement, this module
        does calculations on these and calculates a least
        squares solution for the movement direction, using
        a set of given basis functions.

    """
    def __init__(self, filename, rec_dimensions=(800, 170)):
        """
        
        Arguments:
            filename - first line time steps T and number
                of points in x and y direction given by X
                and Y, followed by T times X times Y lines
                displaying movement in x, y directions.
            rec_dimensions - tuple (x, y), if different
                from standard

        """

        disp = self._read_disp_file(filename)
        X, Y = len(disp[0]), len(disp[0,0])

        self.xs, self.ys = \
                self._define_mesh_pts(X, Y, rec_dimensions)
        self.X, self.Y, self.disp = X, Y, disp


    def _read_disp_file(self, filename):
        """

        Reads the input file, where the file is assumed to
        be a csv file on the form
            T, X, Y
            x0, y0
            x1, y1
            ...
        where we have T x X x Y values, giving the position
        for a given unit for time step t0, t1, ..., for 
        x coordinate x0, x1, ... and y coordinate y0, y1, ...

        x0, y0 gives relative motion of unit (0, 0) at time
        step 0; x1, y1 gives relative motion of unit (1, 0)
        at time step 0, etc.

        Arguments:
            filename - csv file

        Returns:
            4-dimensional numpy array of rec_dimensions
                T x X x Y x 2

        """

        f = open(filename, 'r')
        T, X, Y = map(int, str.split(f.readline(), ","))

        disp = np.zeros((T, X, Y, 2))

        for t in range(T):
            for i in range(X):
                for j in range(Y):
                    d = str.split(f.readline(), ",")[:2]
                    disp[t, i, j] = np.array(d)

        f.close()

        return disp

    def _get_max_timestep(self):
        """

        Finds time step with largest overall movement,
        defined by the combined 2-norm of relaitve
        movement of each unit.

        Returns:
            Integer value in interval [0, T]

        """

        disp = self.disp

        d_max = disp[0]
        d_0 = disp[0]
        max_norm = 0
        max_time_step = 0

        for t in range(len(disp)):
            d = disp[t]
            d_norm = np.linalg.norm(d-d_0)
            if(d_norm > max_norm):
                max_norm = d_norm
                max_time_step = t

        return max_time_step

    def _do_diffusion(self, disp, alpha, N_diff):
        """

        Do diffusion/averaging of values using the molecule
              x
            x o x
              x
        weighting the middle point with value alpha and
        the surrounding points with value (1 - alpha).

        Boundary points are using molecule on the form

            x 0 x
              x

        Arguments:
            disp - numpy array, 2D or 3D
            alpha - weight, value between 0 and 1
            N_diff - number of times to run diffusion

        Returns:
            numpy array of same dimension as disp, with
                averages values as entries

        """
    
        N1, N2 = len(disp), len(disp[0])

        m = lambda a, i, j : alpha*a[i][j] + \
                0.25*(1 - alpha)* \
                (a[max(i-1,0)][j] + a[i][max(j-1,0)] +
                 a[min(i+1,N1-1)][j] + a[i][min(j+1,N2-1)])

        disp1 = disp.copy()
        disp2 = np.zeros_like(disp)

        # do diffusion

        for n in range(N_diff):
            for i in range(N1):
                for j in range(N2):
                    disp2[i][j] = m(disp1, i, j)

            disp1, disp2 = disp2, disp1       #swap pointers

        return disp1 

    def preprocess_data(self, alpha, N_diff):
        """

        Transform raw data to prepared data, to be used
        as input for the least squares solution.

        Arguments:
            alpha - diffusion constant
            N_diff - number of times to do diffusion

        """

        disp, X, Y = self.disp, self.X, self.Y

        # let direction be defined from displacement having
        # largest difference in absolute value

        disp_t = disp[self._get_max_timestep()] - disp[0]
        
        disp_d_t = self._do_diffusion(disp_t, alpha, N_diff)

        # normalize

        for i in range(X):
            for j in range(Y):
                d_norm = np.linalg.norm(disp_d_t[i][j])
                if(d_norm > 1E-10):
                    disp_d_t[i][j] = disp_d_t[i][j]/d_norm
    
        # define to be in first or fourth quadrant

        for i in range(X):
            for j in range(Y):
                if(disp_d_t[i,j,1] < 0):
                    disp_d_t[i,j] = -disp_d_t[i,j]

        org_dir_x, org_dir_y = disp_t[:,:,0], disp_t[:,:,1]
        disp_d_x, disp_d_y = disp_d_t[:,:,0], disp_d_t[:,:,1]

        self.org_dir_x, self.org_dir_y = org_dir_x, org_dir_y
        self.disp_x, self.disp_y = disp_d_x, disp_d_y


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

        xs = np.asarray([rec_dimensions[0]*i/X + dh_x \
                for i in range(X)])
        ys = np.asarray([rec_dimensions[1]*i/Y + dh_y \
                for i in range(Y)])

        return xs, ys

    def find_vector_field(self, M, N, basis_type):
        """
        
        Finds a vector field representing the motion,
        using a least squares solver.

        Arguments:
            M, N - integer values, defines dimensions
                of a two-dimensional function space
            basis_type - defines basis functions, can
                be "trig" or "taylor"

        Returns:
            x and y components of vector field

        """
        
        X, Y, xs, ys, disp_x, disp_y = \
            self.X, self.Y, self.xs, self.ys, \
            self.disp_x, self.disp_y

        l = lsq.Least_sq_solver(X, Y, xs, ys, \
                disp_x, disp_y) 

        VX, VY = l.solve(M, N, basis_type)

        return VX, VY

    def plot_solution(self, filename, title, \
            U, V, arrow=False):
        """

        Gives a quiver plot.

        Arguments:
            filename - save as this file
            title - give title
            U, V - x and y components of vector field
            arrow - boolean value, plot with or
                without arrows

        """

        xs, ys = self.xs, self.ys

        headwidth = 3 if arrow else 0

        plt.subplot(211)
        plt.quiver(xs, ys, \
                np.transpose(U), np.transpose(V), \
                headwidth=headwidth, minshaft=2.5)
        plt.title(title)
        plt.savefig(filename)


if __name__ == "__main__":

    if(len(sys.argv) < 2):
        print("Error: Give input file as argvument")
        exit(-1)

    filename = sys.argv[1]
    
    fd = Fibre_direction(filename)
    fd.preprocess_data(alpha = 0.75, N_diff = 1);

    # plot original data (for specific time step)
        
    org_dir_x, org_dir_y = fd.org_dir_x, fd.org_dir_y
  
    plt.figure(1)
    fd.plot_solution("original.svg", "Original values", \
            org_dir_x, org_dir_y, arrow=True)

    # plot processed data (normalized, positive x values)

    disp_x, disp_y = fd.disp_x, fd.disp_y
    
    plt.figure(2)
    fd.plot_solution("direction.svg", \
            "Original direction (normalized)", \
            disp_x, disp_y)

    # solve linear system, full rank

    M, N = 118, 25
    VX, VY = fd.find_vector_field(M, N, "trig")

    plt.figure(3)
    fd.plot_solution("funspace.svg", \
            "Function space solution (full rank)", \
            VX, VY)

    # solve linear system, reduced rank

    M, N = 50, 12

    VX, VY = fd.find_vector_field(M, N, "trig")
    
    plt.figure(4)
    fd.plot_solution("funspace2.svg", \
            "Function space solution (reduced rank)", \
            VX, VY)

    plt.show()
