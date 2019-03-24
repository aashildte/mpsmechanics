"""

Given data on displacement, this script finds average and overall values
for a number of features.

Run script as (with python 3)

    python analysis.py [input file] [indices]

where the input file is a csv file OR a nd2 file containing
displacement data, and [indices] is a string which indicates which
properties should be recorded, e.g. given as "2 3 5". Quotation marks
required if more than one integer is given. Current supported id's:
    - average beat rate (0)
    - average displacement (1)
    - average x motion (2)
    - average y motion (3)
    - average prevalence (4)
    - average principal strain (5)
    - average principal strain in x direction (x strain) (6)
    - average principal strain in y direction (y strain) (7)

Optionally add
    -p [indices]          (e.g. -p "3 5 6")
for plotting specific properties. Must be a subset of the index set
given above.

---

The *output* of the calculations is saved as 

    [input file name/path].csv

in a folder called 'Analysis' being a subfolder of 'Output'.
This which will be single line, with entries starting with
the identity of the data set; then (in increasing order) the
plotting properties as given above.

---

The (optional) plots will be saved as both a png and svg file,

    [input file name/path] + [attribute].png
    [input file name/path] + [attribute].svg

in a folder called 'Analysis' being a subfolder of 'Figures'.
The attribute corresponds to the properties being plotted.


Åshild Telle / Simula Research Labratory / 2019

"""


import sys
import numpy as np

from optparse import OptionParser

import io_funs as io
import preprocessing as pp
import metrics as mt
import metric_plotting as mp

def get_cl_input():
    """

    Reads command line and transforms into useful variables.

    Returns:
        f_in - filename for displacement data, full path
        idt - prefix of filename, path excluded
        calculation identities - list of integers identifying
            values of interest (which properties to compute)
        plotting identities - list of integers identifying
            values of interest (which properties to plot)

    """

    try:
        f_in = sys.argv[1]
        calc_properties = list(map(int, sys.argv[2].split(" ")))
    except:
        print("Give file name + integers indicationg values of " +
                "interests as arguments (see the README file); " +
                " optionally '-p all' or 'p -disp' for a visual" +
                "output.")
        exit(-1)

    calc_properties.sort()

    # identify data set
    # we assume only one . in the filename itself, and ignore
    # any relative paths (i.e. they are stripped of all .'s)

    de = io.get_os_delimiter()

    idt = f_in.split(de)[-1].split(".")[-2]

    print("Analysing data set: ", idt)

    # optional arguments

    parser = OptionParser()
    parser.add_option("-p")
    (options, args) = parser.parse_args()
    options = vars(options)

    plt_properties = list(map(int, options["p"].split(" ")))
    plt_properties.sort()

    return f_in, idt, calc_properties, plt_properties


def get_plotting_properties(plt_ids, f_in, idt, dimensions, Tmax):
    """

    Defines a dictionary which gives useful information about
    plotting properties; to be forwarded to plotting functions.

    Arguments:
        plt_id - (sorted) list of integers identifying
            which values to plot
        f_in - filename, including full path
        idt - string used for identification of data set
        dimensions - length and height of pictures used for recording
        Tmax - time frame (seconds)

    Return:
        Dictionary with some useful plotting information

    """

    ppl = {}

    # optional argument; default false
    for i in range(8):
        ppl[int(i)] = {"plot" : False}

    for i in plt_ids:
        ppl[i]["plot"] = True
   
    de = io.get_os_delimiter()

    # strip f_in for all relative paths
    while("..") in f_in:
        f_in = de.join(f_in.split(de)[1:])

    subpath = "Figures" + de + "Analysis" + de
    path = subpath + de.join(f_in.split("/")[:-1])
    print("path: ", path)
    io.make_dir_structure(path)

    # get information specificly for metrics
    mp.add_plt_information(ppl, idt, Tmax)

    # other properties
    ppl["path"] = path    
    ppl["dims"] = dimensions     # size of plots over height/length
    ppl["visual check"] = False  # extra plots if applicable

    return ppl


def save_output(idt, calc_idts, values):
    """

    Saves output to file.

    Arguments:
        idt - string with unique identity
        calc_idts - list of identities of interest
        values - list of corresponding output values

    """
    
    # interleave calc_idts, values

    output_vals = []

    for (i, val) in zip(calc_idts, values):
        output_vals.append(i)
        output_vals.append(val)

    values_str = ", ".join([idt] + list(map(str, output_vals))) + "\n"

    de = io.get_os_delimiter()

    subpath = "Output" + de + "Analysis" + de
    path = subpath + de.join(idt.split("/")[:-1])
    io.make_dir_structure(path)

    fout = open(subpath + idt + ".csv", "w")
    fout.write(values_str)
    fout.close()


# set some parameters

alpha = 0.75
N_d = 5
dt = 1./100                          # cl arguments? fps
threshold = 2E-6                     # meters per second
dimensions = (664.30E-6, 381.55E-6)  # picture length/height

# get input parameters

f_in, idt, calc_ids, plt_ids = get_cl_input()

# read + preprocess data

if(".csv" in f_in):
    disp_data, scale = io.read_file_csv(f_in, dimensions[0])
elif(".nd2" in f_in):
    disp_data, scale = io.read_file_nd2(f_in, dimensions[0])
else:
    print("Error: File formate unknown")
    exit(-1)

disp_data = pp.do_diffusion(disp_data, alpha, N_d)

# create dictionary with plotting properties

T = disp_data.shape[0]
Tmax = dt*T

plt_prop = get_plotting_properties(plt_ids, f_in, idt, dimensions, \
        Tmax)

# calculations

values = mt.get_numbers_of_interest(disp_data, calc_ids, \
        scale, dt, plt_prop)

# save as ...

save_output(idt, calc_ids, values)

print("Analysis finished, output saved in Output -> Analysis; " + \
        "plots (if applicable) in Figures -> Analysis.")
