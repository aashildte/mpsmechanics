"""

Given data on displacement, this script finds average and overall values
for a number of features.

Run script as (with python 3)

    python analysis.py file1.csv file2.csv ... -o [output file]

where file1, file2 etc is the given displacement. The fileX part of
the input files are used as an identity for that data set, both used
for plotting (for visual checks) and identity in output file.

Output saved: For each data file given, we calculate and save a set
of values to be used as metrics when comparing the desings. These
will be saved in a file called values.csv, where each row contains
the values
    - identity - file name of the given data set
    - average beat rate
    - average displacement
    - average x motion
    - average prevalence
    - average principal strain

We also plot figures for alignment and beat rate as well as a plot
for each characteristic value. The plots are saved in "Plots", and
each is saved both as a png and as a svg file.


Åshild Telle / Simula Research Labratory / 2018-2019

"""

import sys
import numpy as np

import io_funs as io
import preprocessing as pp
import heart_beat as hb
import mechanical_properties as mc
import metrics as mt

from optparse import OptionParser

def get_underlying_data(f_in, alpha, N_d, idt, dt, xlen, dimensions):
    """

    Preprocessing data + finding maxima.

    Arguments:
        f_in - filename for csv file
        alpha - diffusion parameter
        N_d - diffusion parameter
        idt - identity of data set, for relevant plots
        dt - fps
        xlen - picture x dimension
        dimensions - dimensions of picture used for recording
    
    Returns:
        displacement, numpy array of dimensions T x X x Y x 2
        maxima - local maxima (of each beat)

    """

    disp_data, scale = io.read_disp_file(f_in, xlen)
   
    print("scale given by: ", scale)

    # preprocess data

    disp_data = pp.do_diffusion(disp_data, alpha, N_d)
 
    return disp_data, scale


try:
    assert(len(sys.argv)>1)
except:
    print("Give file names as arguments; optional argument -o output file.")
    exit(-1)


# get output file if given

parser = OptionParser()
parser.add_option("-o")
(options, args) = parser.parse_args()
options = vars(options)

f_o = options["o"]
fout = open(f_o, "w") if f_o is not None else None

alpha = 0.75
N_d = 5
dt = 1./100                       # cl argument? fps

dimensions = (664.30, 381.55)     # cl argument?
threshold = 2E-6                  # meters per second

xlen = (dimensions[0]*1E-6)

output_headers = ",".join([" ", "Average beat rate",
                          "Average displacement",
                          "Average x motion",
                          "Average prevalence",
                          "Average principal strain"])

if fout is not None:
    fout.write(output_headers + "\n")


for f_in in args:
    last_fn = f_in.split("/")[-1].split(".")

    # check suffix - if not a csv file, skip this one

    try:
        prefix, suffix = last_fn
        assert(suffix == "csv")
    except:
        continue

    print("Analyzing data set: " + prefix)

    # perform analysis

    idt = prefix

    disp_data, scale = get_underlying_data(f_in, alpha, N_d, idt, \
            dt, xlen, dimensions)

    T, X, Y = disp_data.shape[:3]

    dx = xlen/X

    values = mt.get_numbers_of_interest(disp_data, scale, idt, dt, \
            dx, dimensions)
    values_str = ", ".join([idt] + list(map(str, values))) + "\n"

    if fout is not None:
        fout.write(values_str)
    
    print("Metrics found: ", values)

if fout is not None:
    fout.close()
