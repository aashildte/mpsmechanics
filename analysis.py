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

import read_data as io
import preprocess_data as pp
import heart_beat as hb
import mechanical_properties as mc


def get_numbers_of_interest(f_in, alpha, N_d, idt, dimensions):
    """

    Arguments:
        f_in - filename
        alpha - diffusion coefficient
        N_d - diffusion number
        idt - data set idt
        dimensions - tuple (x, y), picture dimensions

    Returns:
        average beat rate
        average displacement
        average displacement in x direction
        average prevalence
        max displacement
        max displacement in x direction
        max prevalence

    where each value is taken over peak values only.

    """

    disp_data = io.read_disp_file(f_in)
 
    T, X, Y = disp_data.shape[:3]

    # some parameters

    dt = 1./100            # 100 frames per second
    dx = 2044./664*10E-6   # approximately
    tr = 2*10E-6           # from paper

    T_max = dt*T

    # characteristic values
 
    maxima = np.array(hb.get_beat_maxima(disp_data, idt, T_max))
    e_alpha, e_beta = pp.find_direction_vectors(disp_data, idt, \
        dimensions)

    beat = np.array([(maxima[k] - maxima[k-1]) \
                        for k in range(1, len(maxima))])

    # preprocess data

    disp_data = pp.do_diffusion(disp_data, alpha, N_d)
    disp_data_x = pp.get_projection_vectors(disp_data, e_alpha)

    # get displacement, displacement x, prevalance, principal strain

    disp_data_t = pp.get_overall_movement(disp_data)
    disp_data_x_t = pp.get_overall_movement(disp_data_x)
    
    prevalence = pp.get_overall_movement(\
                     mc.get_prevalence(disp_data, dt, dx, tr))

    pr_strain = pp.get_overall_movement(\
                     mc.compute_principal_strain(disp_data))

    # ... for each beat ...

    maxima = hb.get_beat_maxima(disp_data, idt, T_max)

    # differences:
    beat_r = np.array([maxima[k+1] - maxima[k] \
        for k in range(len(maxima)-1)])

    values = [disp_data_t, disp_data_x_t, prevalence, pr_strain]
    max_vals = np.array([[metric[m] for m in maxima] \
                   for metric in values])

    # plot some values
    suffixes = ["Displacement", "X motion", "Prevalence", \
                    "Principal strain"]

    for (val, s) in zip(values, suffixes):
        hb.plot_maxima(val, maxima, idt, s, T_max)

    # maximum + average

    try:
        max_vals = [max(m) for m in max_vals]
        avg_vals = [np.mean(m) for m in max_vals]
        values = [np.mean(beat_r), max(beat_r)]
    except ValueError:
        print("Empty sequence – no maxima found")


    for k in range(len(max_vals)):
        values.append(max_vals[k])
        values.append(avg_vals[k])

    return values


try:
    assert(len(sys.argv)>1)
except:
    print("Give file names as arguments.")
    exit(-1)

alpha = 0.75
N_d = 5

dimensions = (664.30, 381.55)

output_headers = ",".join([" ", "Average beat rate", "Maximum beat rate", \
                          "Average displacement", "Maximum displacement", \
                          "Average x motion", "Maximum x motion", \
                          "Average prevalence", "Maximum prevalence", \
                          "Average principal strain", "Maximum principal strain"])

fout = open("values.csv", "w")

for f_in in sys.argv[1:]:
    last_fn = f_in.split("/")[-1].split(".")

    # check suffix - if not a csv file, skip this one

    prefix, suffix = last_fn

    if(suffix != "csv"):
        continue

    # perform analysis

    idt = prefix

    values = get_numbers_of_interest(f_in, alpha, N_d, idt, dimensions)
    values_str = ", ".join([idt] + list(map(str, values)))

    fout.write(values_str)

fout.close()
