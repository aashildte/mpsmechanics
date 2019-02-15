"""

Given data on displacement, this script finds average and overall values
for a number of features.

Run script as (with python 3)

    python analysis.py file1.csv file2.csv ...

where file1, file2 etc is the given displacement. The fileX part of
the input files are used as an identity for that data set, both used
for plotting (for visual checks) and identity in output file.

Output saved: For each data file given, we calculate and save a set
of values to be used as metrics when comparing the desings. These
will be saved in a file called values.csv, where each row contains
the values
    - identity - file name of the given data set
    - average beat rate
    - maximum beat rate
    - average displacement
    - maximum displacement
    - average x motion
    - maximum x motion
    - average prevalence
    - maximum prevalence
    - average principal strain
    - maximum principal strain

We also plot figures for alignment and beat rate as well as a plot
for each characteristic value. The plots are saved in "Plots", and
each is saved both as a png and as a svg file.


Åshild Telle / Simula Research Labratory / 2019

"""

import sys
import numpy as np

import io_funs as io
import preprocessing as pp
import heart_beat as hb
import mechanical_properties as mc
import metrics as mt

def get_underlying_data(f_in, alpha, N_d, idt, dt, dimensions):
    """

    Preprocessing data + finding maxima.

    Arguments:
        f_in - filename for csv file
        alpha - diffusion parameter
        N_d - diffusion parameter
        idt - identity of data set, for relevant plots
        dt - fps
        dimensions - dimensions of picture used for recording
    
    Returns:
        displacement, numpy array of dimensions T x X x Y x 2
        maxima - local maxima (of each beat)

    """

    disp_data = io.read_disp_file(f_in)
    
    # preprocess data

    disp_data = pp.do_diffusion(disp_data, alpha, N_d)
 
    # beat rate - find all maximum points

    T, X, Y = disp_data.shape[:3]
    T_max = dt*T
    maxima = np.array(hb.get_beat_maxima(disp_data, idt, T_max))

    return disp_data, maxima


try:
    assert(len(sys.argv)>1)
except:
    print("Give file names + output file as arguments.")
    exit(-1)

f_out = sys.argv[-1]

alpha = 0.75
N_d = 5
dt = 1./100            # 100 frames per second

dimensions = (664.30, 381.55)

output_headers = ",".join([" ", "Average beat rate", "Maximum beat rate", \
                          "Average displacement", "Maximum displacement", \
                          "Average x motion", "Maximum x motion", \
                          "Average prevalence", "Maximum prevalence", \
                          "Average principal strain", "Maximum principal strain"])

fout = open(f_out, "w")

fout.write(output_headers + "\n")

for f_in in sys.argv[1:-1]:
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

    disp_data, maxima = get_underlying_data(f_in, alpha, N_d, idt, dt, dimensions)

    values = mt.get_numbers_of_interest(disp_data, maxima, idt, dimensions)
    values_str = ", ".join([idt] + list(map(str, values))) + "\n"

    fout.write(values_str)

fout.close()
