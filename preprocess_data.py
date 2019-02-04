"""

Given experimental data on movement, these functions preprocess the given values.
 - performs diffusion
 - finds main direction of detected intensity


Åshild Telle / Simula Research Labratory / 2019

"""


import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

import read_data as io


def perform_operation(A, fn, shape=None):
    """

    Performs an operation on all values of A, where A is assumed to be
    a T x X x Y x 2 numpy array.

    Arguments:
        A - numpy array
        fn - function f((x, y), i, j) -> (x, y)
        shape - optional, like A if not specified

    Returns:
        T x X x Y x 2, numpy array with resulting values

    """

    T, X, Y = A.shape[:3]

    if(shape is None):
        shape = A.shape
    
    disp = np.zeros(shape)

    # perform operation, given as a function

    for t in range(T):
        for x in range(X):
            for y in range(Y):
                disp[t, x, y] = fn(A[t, x, y], x, y)

    return disp


def get_overall_movement(data):
    """

    Finds sum of L2 norm of each vector,
    
        n(t) = sum(i, j) sqrt(x_{t,i,j}^2 + y_{t,i,j}^2)

    for t = 0 ... T.

    Arguments:
        Data - numpy array, of dimensions T x X x Y x 2

    Returns:
        Sum array - numpy array of dimension T  

    """
    T, X, Y = data.shape[:3]

    disp_norm = np.zeros(T)
    
    for t in range(T):
        disp_norm[t] = sum(np.linalg.norm(data[t].reshape(X*Y, 2), axis=1))

    return disp_norm


def get_max_ind(disp):
    """

    Finds the index of the maximum value of disp. If there are multiple
    maxima (same number) the first one will be returned.

    Arguments:
        disp - 1D list-like data structure

    Returns
        index corresponding to maximum index

    """

    T = len(disp)

    max_t = (0, disp[0])

    for t in range(T):
        if(disp[t] > max_t[1]):
            max_t = (t, disp[t])

    return max_t[0]

def find_direction_vectors(disp, idt, dimensions, mu=1E-5):
    """

    From the given displacement, this function finds the
    direction of most detected movement using linear regression.

    Calls a function which plots the values along with direction vectors
    for visual check.

    Arguments:
        disp - T x X x Y x 2 dimensional numpy array
        dimensions - pair of dimension values (x, y)
        mu - optional argument, threshold for movement detection

    Returns:
	e_alpha - vector in 1st or 4th quadrant along most movement
	e_beta  - e_alpha rotatet pi/2 anti-clockwise

    """
    T, X, Y = disp.shape[:3]

    xs = []
    ys = []

    for t in range(T): 
        for x in range(X):
            for y in range(Y):
                n_vector = disp[t, x, y, :]
                if(np.linalg.norm(n_vector) >= mu):
                    xs.append(x)
                    ys.append(y)

    slope = st.linregress(xs, ys)[0]

    dir_v = np.array([1, slope])

    e_alpha = np.linalg.norm(dir_v)*dir_v
    e_beta  = np.array([-e_alpha[1], e_alpha[0]])

    _plot_data_vectors(xs, ys, X, Y, e_alpha, e_beta, idt, dimensions)

    return e_alpha, e_beta


def _plot_data_vectors(xs, ys, X, Y, e_alpha, e_beta, idt, dimensions):
    """

    Plots data points along with direction vectors.

    Figures saved as Plots/[idt]_alignment.png, Plots/[idt]_alignment.svg

    Arguments:
        xs - data points along x axis
        ys - data points along y axis
        X  - number of data points in x direction
        Y  - number of data points in y direction
        e_alpha - main direction
        e_beta  - perpendicular vector
        idt - idt for plots
        dimensions - pair of dimensions (x, y)

    """
    # scale dimensions to standard size in x direction

    scale = 6.4/dimensions[0]
    dimensions_scaled = (scale*dimensions[0], scale*dimensions[1])

    plt.figure(figsize=dimensions_scaled)
    plt.xlabel('$\mu m$')
    plt.ylabel('$\mu m$')

    # downsample data for plotting - only plot each value once

    pairs = list(set([(x, y) for (x, y) in zip(xs, ys)]))

    p_x = np.array([p[0] for p in pairs])
    p_y = np.array([p[1] for p in pairs])

    # scale

    p_x = dimensions[0]/X*p_x
    p_y = dimensions[1]/Y*p_y

    path = "Plots"
    io.make_dir_structure(path)

    plt.scatter(p_x, p_y, color='gray')

    sc = [0.1*max(p_x), 0.1*max(p_y)]

    for e in [e_alpha, e_beta]:
        plt.plot([0, sc[0]*e[0]], [0, sc[1]*e[1]], color='red')

    plt.savefig(path + "/" + idt + "_alignment.png")
    plt.savefig(path + "/" + idt + "_alignment.svg")

    plt.figure()
    plt.clf()

def get_projection_vectors(data, e_i):
    """

    Extracts the parallel part of each component in disp,
    with respect to given vector e_i (does a projection).

    Arguments:
        e_i - non-zero vector (numpy array/two values)

    Returns:
        T x X x Y x 2 numpy array with parallel components

    """
    e_i = 1./(np.linalg.norm(e_i))*e_i

    f = lambda x, i, j, e_i=e_i: np.dot(x, e_i)*e_i

    return perform_operation(data, f)


def get_min_max_scale(dir_vector, data):
    """
    TODO - code to find rectangle
    """

    T, X, Y = data.shape[:3]

    Cs = np.array([0, 0]), np.array([0, Y-1]), \
       np.array([X-1, 0]), np.array([X-1, Y-1])

    S = [np.dot(dir_vector, c) for c in Cs]
    return min(S), max(S)


def find_mean_movement_pt(dir_vector, data):
    """
    TODO - code to find rectangle
    """

    T, X, Y = data.shape[:3]

    N = 1000
    movement_bins = np.zeros(N)

    S_min, S_max = self.get_min_max_scale(dir_vector, data)
    S = S_max - S_min

    for x in range(X):
        for y in range(Y):
            b = np.array([x, y])
            s = abs(np.dot(dir_vector, b))
            n = int((N-1)*s/S)               # map to a bin
	
            for t in range(T):
                n_vector = disp[t, x, y, :]
                proj_val = -S_min + np.dot(n_vector, dir_vector)
                movement_bins[n] = movement_bins[n] + proj_val

    accum_values = np.zeros(N)
    s_val = 0

    for n in range(N):
        s_val = s_val + movement_bins[n]
        accum_values[n] = s_val

    average_movement = np.mean(accum_values)

    for n in range(N):
        if(average_movement < accum_values[n]):
            break

    # idea: last point * mean n / number of possible bins
    scaling_factor = (s*n)/N

    return scaling_factor*dir_vector

def find_movement_rectangle(data, e_alpha, e_beta):
    """
        TODO - code to find rectangle
    """

    T, X, Y = data.shape[:3]

    e1, e2 = np.array([1, 0]), np.array([0, 1])

    s1 = self.find_mean_movement_pt(e_alpha)       
    s2 = self.find_mean_movement_pt(e_beta)       

    x = np.dot(e1, s1)
    y = np.dot(e2, s2)

    print("x, y: ", x, y)     # should be central point


def do_diffusion(data, alpha, N_diff):
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
    
    or

        x 0
          x

    TODO - rewrite to utilize perform_operation function

    Arguments:
        data - numpy array of dimensions T x X x Y x 2
        alpha - weight, value between 0 and 1
        N_diff - number of times to run diffusion

    Returns:
        T x X x Y x 2 numpy array, data after averaging process

    """

    T, X, Y = data.shape[:3]

    # keep original array untouched

    disp = data.copy()

    # diffusion function

    m = lambda a, i, j : alpha*a[i][j] + 0.25*(1 - alpha)* \
             (a[max(i-1,0)][j] + a[i][max(j-1,0)] +
             a[min(i+1,X-1)][j] + a[i][min(j+1,Y-1)])

    # do diffusion

    for t in range(T):
        disp1 = disp[t]                  # copy
        disp2 = np.zeros_like(disp1)

        for n in range(N_diff):
            for i in range(X):
               for j in range(Y):
                   disp2[i][j] = m(disp1, i, j)

            disp1, disp2 = disp2, disp1       # swap pointers

        disp[t] = disp1                  # overwrite (if not equal)

    return disp


def normalize_values(data):
    """

    Normalises each non-zero vector.

    Arguments:
        data - T x X x Y x 2 numpy array, original values

    Returns:
        T x X x Y x 2 numpy array, normalized values

    """

    f = lambda x, i, j: (1/np.linalg.norm(x))*x \
            if np.linalg.norm(x) > 1E-10 else np.zeros(2)

    return perform_operation(data, f)


def flip_values(data):
    """

    Rotate each vector, to first or fourth quadrant

    Arguments:
        data - T x X x Y x 2 numpy array, original values

    Returns:
        T x X x Y x 2 numpy array, flipped values

    """

    f = lambda x, i, j: -x if x[1] < 0 else x
    
    return perform_operation(data, f)


if __name__ == "__main__":

    try:
        f_in = sys.argv[1]
        idt = sys.argv[2]
    except:
        print("Error reading file names. Give displacement file name as " + \
                  "first argument, identity as second.")
        exit(-1)
    
    data = io.read_disp_file(f_in)

    # unit tests:
    
    T, X, Y = data.shape[:3]
    D = (T, X, Y, 2)
   
    assert(perform_operation(data, lambda x, i, j : 1).shape==D) 
    print("Perform operation check passed")

    e_a, e_b = find_direction_vectors(data, idt)

    assert(e_a.shape==(2,) and e_b.shape==(2,))
    print("Vector check passed")
    
    assert(get_projection_vectors(data, e_a).shape==D)
    print("Projection vectors check passed")
    
    # TODO check for get_min_max_scale

    # TODO check for find_mean_movement_pt

    # TODO check for find_movement_rectangle
    
    assert(do_diffusion(data, 0.75, 2).shape==D)
    print("Diffusion check passed")

    assert(normalize_values(data).shape==D)
    print("Normalization check passed")

    assert(flip_values(data).shape==D)
    print("Flip values check passed")

    print("All checks passed.")
