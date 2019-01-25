"""

Given data on displacement, this script finds average and overall values
for a number of features.

Run script as (with python 3)

    python analyzis.py file1.csv file2.csv ...

where file1, file2 etc is the given displacement. The fileX part of the
input files are used as an identity for that data set, both used for
plotting (for visual checks) and identity in output file.

Output saved: For each idt,
    maxima_idt.png, maxima_idt.svg
    alignment_idt.png, alignment_idt.svg
    values.csv on the form

        , Average beat rate, Average displacement, ...
    idt , average beat rate for idt, average displacement for idt, ...

    where each row corresponds to a given input file.


Aashild Telle / Simula Research Labratory / 2019

"""

import sys
import numpy as np

import read_data as io
import preprocess_data as pp
import heart_beat as hb
import mechanical_properties as mc


def get_numbers_of_interest(f_in, alpha, N_d, idt):
    """

    Arguments:
        f_in - filename
        alpha - diffusion coefficient
        N_d - diffusion number
        idt - data set idt

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
 
    T, X, Y = pp.get_dimensions(disp_data)

    # some parameters

    dt = 1./100            # 100 frames per second
    dx = 2044./664*10E-6   # approximately
    tr = 2*10E-6           # from paper

    T_max = dt*T

    # characteristic values
 
    maxima = np.array(hb.get_beat_maxima(disp_data, idt, T_max))
    e_alpha, e_beta = pp.find_direction_vectors(disp_data, idt)

    avg_beat = hb.get_average(maxima)*dt    # in Hz

    # preprocess data

    disp_data = pp.do_diffusion(disp_data, alpha, N_d)
    disp_data_x = pp.get_projection_vectors(disp_data, e_alpha)

    # get displacement, displacement x, prevalance

    disp_data_t = pp.get_overall_movement(disp_data)
    disp_data_x_t = pp.get_overall_movement(disp_data_x)
    
    prevalence = pp.get_overall_movement(\
                     mc.get_prevalence(disp_data, dt, dx, tr))
    # ... for each beat ...

    maxima = hb.get_beat_maxima(disp_data, idt, T_max)

    max_contr_L2 = np.array([disp_data_t[m] for m in maxima])
    max_contr_x = np.array([disp_data_x_t[m] for m in maxima])
    max_prev = np.array([prevalence[m] for m in maxima])

    # maximum + average

    max_vals = [max(v) for v in [max_contr_L2, max_contr_x, max_prev]]
    avg_vals = [np.mean(v) for v in [max_contr_L2, max_contr_x, max_prev]]

    return [avg_beat] + max_vals + avg_vals


try:
    assert(len(sys.argv)>1)
except:
    print("Give file names as arguments.")
    exit(-1)

alpha = 0.75
N_d = 5

output_headers = ", Average beat rate, Average displacement, Average x displacement, Average prevalence, Max displacement, Max x displacement, Max prevalence"

fout = open("values.csv", "w")

for f_in in sys.argv[1:]:
    idt = f_in.split("/")[-1].split(".")[0]

    values = get_numbers_of_interest(f_in, alpha, N_d, idt)
    values_str = ", ".join([idt] + list(map(str, values)))

    fout.write(values_str)

fout.close()
