"""

Given data on displacement, this script finds average and overall values
for a number of features.

Run script as (with python 3)

    python analysis.py [input file] [indices]

where the input file is a csv file containing displacement data, and
[indices] is a string which indicates which properties should be
recorded, e.g. given as "2 3 5". Quotation marks required if more
than one integer is given. Current supported id's:
    - average beat rate (0)
    - average displacement (1)
    - average x motion (2)
    - average y motion (3)
    - average prevalence (4)
    - average principal strain (5)
    - average principal strain in x direction (x strain) (6)
    - average principal strain in y direction (y strain) (7)

Optionally add
    -p all
for plotting all possible related plots (extensive visual check), or 
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


def get_plotting_properties(options, idt, dimensions, Tmax):

    ppl = {}

    # optional argument; default false
    for i in range(8):
        ppl[int(i)] = {"plot" : False}

    if(options["p"] is not None):
        if(options["p"]=="all"):
            ppl["all"] = True
            for i in range(8):
                ppl[int(i)]["plot"] = True
        else:
            for i in options["p"].split(" "):
                ppl[int(i)]["plot"] = True

    mt.add_plt_information(ppl, idt, Tmax)

    # other properties
    
    ppl["dims"] = dimensions
    ppl["visual check"] = False

    return ppl

def save_output(idt, values):
    values_str = ", ".join([idt] + list(map(str, values))) + "\n"

    de = io.get_os_delimiter()

    subpath = "Output" + de + "Analysis" + de
    path = subpath + de.join(idt.split("/")[:-1])
    io.make_dir_structure(path)

    fout = open(subpath + idt + ".csv", "w")
    fout.write(values_str)
    fout.close()    


try:
    f_in = sys.argv[1]
    calc_properties = list(map(int, sys.argv[2].split(" ")))
except:
    print("Give file name + integers indicationg values of " +
            "interests as arguments (see the README file); " +
            " optionally '-p all' or 'p -disp' for a visual output.")
    exit(-1)

calc_properties.sort()

# identify data set
# we assume only one . in the filename itself, and ignore
# any relative paths (i.e. they are stripped of all .'s)

idt = f_in.split(".")[-2]

print("Analysing data set: ", idt)

# optional arguments

parser = OptionParser()
parser.add_option("-p")
(options, args) = parser.parse_args()
options = vars(options)

# set some parameters

alpha = 0.75
N_d = 5
dt = 1./100                       # cl argument? fps
threshold = 2E-6                  # meters per second
dimensions = (664.30, 381.55)     # cl argument?
xlen = (dimensions[0]*1E-6)

# read + preprocess data

disp_data, scale = io.read_disp_file(f_in, xlen)
disp_data = pp.do_diffusion(disp_data, alpha, N_d)

T, X, Y = disp_data.shape[:3]

# create dictionary with plotting properties

Tmax = dt*T
plt_prop = get_plotting_properties(options, idt, f_in, Tmax)

# calculations

values = mt.get_numbers_of_interest(disp_data, calc_properties, \
        scale, dt, plt_prop)

# save as ...

save_output(idt, values)

print("Analysis finished, output saved in Output -> Analysis; " + \
        "plots (if applicable) in Figures -> Analysis.")
