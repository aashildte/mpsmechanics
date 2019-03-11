
"""

Tracks given points for a given data set over time.

Run as

    python track_points.py [filename displacement] [filename points]

where the displacement file is a csv file of displacement coordinates
(starting with T, X, Y; then a list of T x X x Y x 2 displacement
values, one for each T x X x Y coordinate, in x and one in y direction;
and the points are given as one pair per line, as in

    x0 y0
    x1 y1

    ...


Need to discuss output values.

Åshild Telle / Simula Research Labratory / 2019

"""


import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

import io_funs as io
import preprocessing as pp


def read_pt_file(f_in):
    """

    Reads in values for pillar coordinates + radii.

    Arguments:
        f_in - Filename

    Returns;
        Numpy array of dimensions P x 3, P being the number of
            points; entries being x, y, radius for all points.

    """

    f = open(f_in, "r")

    lines = f.read().split("\n")[1:]

    if(len(lines[-1])==0):
        lines = lines[:-1]      # ignore last line if empty

    p_values = [[float(x) for x in line.split(",")] \
                        for line in lines]

    # flip x and y; temporal solution due to two different conventions
    # used. TODO - change this into one convention


    for i in range(len(p_values)):
        x, y, r = p_values[i]
        p_values[i] = [y, x, r]

    p_values = np.array(p_values)

    f.close()

    return p_values


def define_pillars(p_values, N=100):
    """

    Transforms x, y, r values to mesh values.

    Arguments:
        p_values - list like structure of dimensions P x 3
        N - number of mesh points used

    Returns:
        p_values - mesh points on the circumsphere of the circle
            defined by x, y, z

    """

    P = len(p_values)
    pillars = np.zeros((P, N, 2))

    angles = np.linspace(0, 2*np.pi, N)

    for i in range(P):
        x, y, r = p_values[i]
        for j in range(N):
            pillars[i, j, 0] = x + r*np.cos(angles[j])
            pillars[i, j, 1] = y + r*np.sin(angles[j])

    return pillars


 

def track_pillars_over_time(data, pillars, dimensions, path):
    """

    Tracks position of mesh poinds defined for pillars over time, based
    on the displacement data.

    Arguments:
        data - displacement data, numpy array of size T x X x Y x 2
        pillars - mesh points
        path - save output values here

    """

    de = io.get_os_delimiter()

    T, X, Y = data.shape[:3]
    P, N = pillars.shape[:2]

    xs = np.linspace(0, dimensions[0], X)
    ys = np.linspace(0, dimensions[1], Y)

    new_values = np.zeros((T, P, N, 2))

    xscale = dimensions[0]/X
    yscale = dimensions[1]/Y
    
    scale = 6/dimensions[0]
    dims_scaled = (dimensions[0]*scale, dimensions[1]*scale)

    x_pos = np.zeros(T)
    y_pos = np.zeros(T)

    x_org = np.zeros(T)
    y_org = np.zeros(T)

    x_ind = int(X/4)
    y_ind = int(Y/4)

    track_pt = np.array((xs[x_ind], ys[y_ind]))

    for t in range(T):
        Xs = xscale*data[t,:,:,0].transpose()  #???? check with someone
        Ys = yscale*data[t,:,:,1].transpose()  #????

        fn_x = interpolate.interp2d(xs, ys, Xs, kind='cubic')
        fn_y = interpolate.interp2d(xs, ys, Ys, kind='cubic')
        
        fn = lambda x, y: 0*np.array([x, y]) + \
            np.array([fn_x(x, y)[0], fn_y(x, y)[0]])

        # check
        """
        for x in range(X):
            for y in range(Y):
                z = fn(xs[x], ys[y])
                print("compare: ", data[t, x, y] + np.array([xs[x], ys[y]]), z)
        """
        for p in range(P):
            for n in range(N):
                new_values[t, p, n] = fn(pillars[p, n, 0], pillars[p, n, 1])

        x_vals = new_values[t, :, :, 0].flatten()
        y_vals = new_values[t, :, :, 1].flatten()

        
        fg = plt.figure(t, figsize=dims_scaled)
        t_id = "%04d" % t
        
        plt.scatter(x_vals, y_vals, s=0.01)
        plt.savefig(path + de + "state_" + t_id + ".png")

        plt.close()
        

        x_org[t] = data[t, x_ind, y_ind, 0]
        y_org[t] = data[t, x_ind, y_ind, 1]

        z = fn(*track_pt)
        x_pos[t] = z[0]
        y_pos[t] = z[1]

    # for output check - TODO remove once we know it works

    ts = np.linspace(0, T, T)

    plt.figure(0)
    plt.plot(ts, x_pos, ts, y_pos)
    plt.legend(["New x", "New y"])

    plt.figure(1)
    plt.plot(ts, x_org, ts, y_org)
    plt.legend(["Org x", "Org y"])

    plt.show()

    return new_values
    

try:
    f_in1 = sys.argv[1]
    f_in2 = sys.argv[2]
except:
    print("Give displacement file and position file as first and second " +
            "positional argument.")
    exit(-1)


unit = 1E-6
dimensions = unit*np.array((664.30, 381.55))
xlen=dimensions[0]

data, scale = io.read_disp_file(f_in1, xlen)
data = scale*data
#data = data[:5]
data = pp.do_diffusion(data, 0.75, 5, over_time=True)

#ip_values = calc_ip_values(scale*data, unit*dimensions)

points = unit*read_pt_file(f_in2)

pillars = define_pillars(points)

de = io.get_os_delimiter()
path = "Figures" + de + "Track points"
io.make_dir_structure(path)

track_pillars_over_time(data, pillars, dimensions, path)
