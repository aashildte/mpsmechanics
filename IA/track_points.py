
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

import io_funs as io
import preprocessing as pp


def read_pt_file(f_in):

    f = open(f_in, "r")

    lines = f.read().split("\n")

    if(len(lines[-1])==0):
        lines = lines[:-1]      # ignore last line if empty


    points = np.array([[float(x) for x in line.split(", ")] \
                        for line in lines])

    f.close()

    return points

def find_mesh_pts(data, dimensions):

    X, Y = data.shape[1:3]

    xs = np.linspace(0, dimensions[0], X)
    ys = np.linspace(0, dimensions[1], Y)

    return np.array([xs, ys])


def find_nbhs(xs, pt):
    """
    
    Finds indices of x point which is just below pt.

    Arguments:
        xs - list of mesh points in a given direction
        pt - float, assumed to be > xs[0] and < xs[-1]

    Returns:
        i - index of mesh point just *below* pt

    """

    i = 0

    while(i < (len(xs)-1) and xs[i+1] < pt):
        i = i+1
 
    return i
 

def track_point_over_time(data, pt, xs, ys):
    
    i = find_nbhs(xs, pt[0])
    j = find_nbhs(ys, pt[1])

    T, X, Y = data.shape[:3]

    x_disp = np.zeros(T)
    y_disp = np.zeros(T)

    w1 = (pt[0] - xs[i])/(xs[i+1] - xs[i])
    w2 = (pt[1] - ys[j])/(ys[j+1] - ys[j])

    for t in range(T):
        x = (1 - w2)*((1 - w1)*data[t, i, j, 0] + \
                            w1*data[t, i+1, j, 0]) + \
                 w2*((1 - w1)*data[t, i, j+1, 0] + \
                         w1*data[t, i+1, j+1, 0])
        
        y = (1 - w2)*((1 - w1)*data[t, i, j, 1] + \
                           w1*data[t, i+1, j, 1]) + \
                 w2*((1 - w1)*data[t, i, j+1, 1] + \
                           w1*data[t, i+1, j+1, 1])

        x_disp[t] = x
        y_disp[t] = y

    ts = np.linspace(0, T, T)

    plt.figure()

    plt.plot(ts, x_disp, ts, y_disp)
    plt.legend(["xs", "ys"])
    plt.xlabel("Time")

try:
    f_in1 = sys.argv[1]
    f_in2 = sys.argv[2]
except:
    print("Give displacement file and position file as first and second " +
            "positional argument.")
    exit(-1)


dimensions = np.array((664.30, 381.55))
unit = 1E-6
xlen=dimensions[0]*unit
data, scale = io.read_disp_file(f_in1, xlen)
data = pp.do_diffusion(data, 0.75, 5)

xs, ys = find_mesh_pts(scale*data, unit*dimensions)

points = unit*read_pt_file(f_in2)

for pt in points:
    track_point_over_time(data, pt, xs, ys)

plt.show()
