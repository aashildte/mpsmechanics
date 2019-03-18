
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

Optionally add the argument --plot to get a visual output.

Output given as
    1) a set of files on the form pt[X]_[Y].csv, saved in where the
       *first* line give number of time steps, number of tracking
       points for that pillar:
           T, N
       (which indicates N uniformly distributed points around (X, Y)
       and then T*N lines of the form
           x y
       which gives the position of the n'th point at time step t;
       all positions first listed for time step 0, then for time
       step 1, etc.
    2) one file called at_maxima.csv, where the first line give
       the maximum indices:
           _, _ , m1, _ , m2, _, m3, _, ...
       and the remaning lines contains the following values:
           original x position of pillar
           original y position of pillar, 
           x position at index m1
           y position at index m1
           x position at index m2
           y position at index m2 
           ...

All files for numerical output are saved in
    Output -> Track pillars -> File idt

If --plot is given as an argument, we're plotting the values at each
time step, saving them as state_[t].png, as well as the x and y
position over all time step, as an average of all points along the
circle representing the pillar – so one plot per pillar. These
are saved as
    all_time_steps_x.png
    all_time_steps_y.png

All plots are saved in are saved in
    Figures -> Track pillars -> File idt

Åshild Telle / Simula Research Labratory / 2019

"""


import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

import io_funs as io
import preprocessing as pp
import heart_beat as hb


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

    p_values = [[int(x) for x in line.split(",")] \
                        for line in lines]  # or float???

    # flip x and y; temporal solution due to two different
    # conventions used. TODO - use same everywhere

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


def interpolate_step(xs, ys, org_data):
    """

    Interpolates given data; defines functions based on this.
    First function gives relative displacement; second absolute.

    Arguments:
        xs - x coordinates
        ys - y coordinates
        org_data - displacement data; X x Y x 2 numpy array

    Returns:
        function f : R2 - R2 - relative
        function g : R2 - R2 - absolute

    """

    Xs = org_data[:,:,0].transpose()
    Ys = org_data[:,:,1].transpose()

    fn_x = interpolate.interp2d(xs, ys, Xs, kind='cubic')
    fn_y = interpolate.interp2d(xs, ys, Ys, kind='cubic')
    
    fn1 = lambda x, y: np.array([fn_x(x, y)[0], fn_y(x, y)[0]])
    fn2 = lambda x, y: np.array([x, y]) + fn1(x, y)
   
    return fn1, fn2


def plot_x_y_coordinates(values, dimensions, t, path):
    """

    Scatter plot of coordinates of each tracking point.

    Arguments:
        values to be plotted - three-dimensional numpy array,
            last dimensions being 2 (x and y)
        dimensions - picture dimensions, to match output plot
            with actual picture dimensions
        t - timestep, used for identity
        path - save figures here

    """

    scale = 6/dimensions[0]
    dims_scaled = (dimensions[0]*scale, dimensions[1]*scale)
    de = io.get_os_delimiter()

    x_vals = values[:, :, 0].flatten()
    y_vals = values[:, :, 1].flatten()

    fg = plt.figure(t, figsize=dims_scaled)
    t_id = "%04d" % t
    plt.xlim(0, dimensions[0])
    plt.ylim(0, dimensions[1])

    plt.scatter(x_vals, y_vals, s=0.01)
    plt.savefig(path + de + "state_" + t_id + ".png", dpi=300)

    plt.close()


def write_all_values_to_file(all_values, coords, path):
    """

    Output to files: T x N values for each pillar

    Arguments:
        all_values - numpy array of dimension T x P x N x 2
        coords - midpoints; numpy array of dimension P x 2
        path - save here

    """

    T, P, N = all_values.shape[:3]

    for p in range(P):
        coords = mpoints[p]
        filename = path + de + "pillar_" + str(coords[0]) + "_" + \
                str(coords[1]) + ".csv"

        f = open(filename, "w")

        f.write(str(T) + ", " + str(N) + "\n")

        for t in range(T):
            for n in range(N):
                x, y = all_values[t, p, n]
                f.write(str(x) + ", " + str(y) + "\n")

        f.close()


def plot_over_time(values, coords, path):
    """

    Plots tracked values over time.

    Arguments:
        values - numpy array of dimension T x P x 2
        coords - coordinates; (x, y) for P pillars
        path - save figures here

    """
    
    T, P = values.shape[:2]
    fps = 100
    Tmax = T/fps
    ts = np.linspace(0, Tmax, T)
    
    de = io.get_os_delimiter()
    filenames = [path + de + "all_time_steps_x.png", \
            path + de + "all_time_steps_y.png"]
    titles = ["Displacement over time (x values)", \
              "Displacement over time (y values)"]

    for i in range(2):
        for p in range(P):
            plt.plot(ts, values[:,p,i])

        plt.legend([str(x[i]) for x in coords])
        plt.savefig(filenames[i], dpi=300) 
        plt.close()


def write_max_disp_to_file(mid_values, max_indices, coords, path):
    """

    Writes values at maximum displacement to a file.

    Arguments:
        mid_values - T x P x 2 numpy array
        max_indices - list alike structure for indices of maxima
        coords - coordinates of midpoints
        path - save file here

    """

    de = io.get_os_delimiter()
    filename = path + de + "at_maxima.csv"
    f = open(filename, "w")

    max_str = ", ," + ", , ".join([str(m) \
            for m in max_indices]) + "\n"
    f.write(max_str)
    
    for p in range(len(coords)):
        m_values = [mid_values[m,p] for m in max_indices]

        pos_str = str(coords[p,0]) + ", " + str(coords[p,1]) + \
                ", " + ", ".join([str(m[0]) + ", " + str(m[1]) \
                for m in m_values]) + "\n"

        f.write(pos_str)

    f.close()


def track_pillars_over_time(data, pillars, mpoints, dimensions, path_o, \
        path_p=None, plot_values=False):
    """

    Tracks position of mesh poinds defined for pillars over time, based
    on the displacement data.

    Arguments:
        data - displacement data, numpy array of size T x X x Y x 2
        pillars - mesh points; on a circle around each middle point
        mpoints - mesh points; center of each pillar
        path_o - save output values here
        path_p - save plots here
        plot_values - boolean - plot values or not

    """

    # some general values

    T, X, Y = data.shape[:3]
    P, N = pillars.shape[:2]

    xs = np.linspace(0, dimensions[0], X)
    ys = np.linspace(0, dimensions[1], Y)
    
    # for tracking pillars
    
    all_values = np.zeros((T, P, N, 2))

    # for force measurements

    mid_values = np.zeros((T, P, 2))

    for t in range(T):

        fn1, fn2 = interpolate_step(xs, ys, data[t])
        
        # all points

        for p in range(P):
            for n in range(N):
                all_values[t, p, n] = fn2(pillars[p, n, 0], \
                        pillars[p, n, 1])

        if(plot_values):
            plot_x_y_coordinates(all_values[t], dimensions, t, path_p)

        # midpoints
        for p in range(P):
            mid_values[t, p] = fn1(*mpoints[p])

    # save values

    write_all_values_to_file(all_values, mpoints, path_o)

    # find values at maximum displacement
    max_indices = hb.calc_beat_maxima_2D(data)

    write_max_disp_to_file(mid_values, max_indices, mpoints, path)
    
    if(plot_values):
        plot_over_time(mid_values, mpoints, path_p)

try:
    f_in1 = sys.argv[1]
    f_in2 = sys.argv[2]

    plot_values = "--plot" in sys.argv
except:
    print("Give displacement file and position file as first and" +
            "second positional argument.")
    exit(-1)


dimensions = np.array((1990, 958))
xlen=dimensions[0]

data, scale = io.read_disp_file(f_in1, xlen)

# scale back to "picture" units

dx = xlen/data.shape[1]
data = (scale/dx)*data

T, X, Y = data.shape[:3]

# averaging process - adjustable values
alpha = 0.75
N_diff = 5
data = pp.do_diffusion(data, alpha, N_diff, over_time=True)

# points of interest

points = read_pt_file(f_in2)
pillars = define_pillars(points)

# setup for saving things

de = io.get_os_delimiter()
idt = f_in1.split(de)[-1].split(".")[0]

path_p = "Figures" + de + "Track points" + de + idt
path_o = "Output" + de + "Track points" + de + idt

for path in (path_p, path_o):
    io.make_dir_structure(path)

# track values

mpoints = np.array([(x, y) for (x, y, r) in points])

track_pillars_over_time(data, pillars, mpoints, dimensions, path_o, \
        plot_values=plot_values, path_p=path_p)
