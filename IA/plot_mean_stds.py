"""

Given output metrics, this script gathers and evaluates these.

Run script as (with python 3)

    python plot_mean_stds.py file1.csv file2.csv ...

where file1, file2 etc are (combined) output values from the
analysis.py script, i.e. csv files containing lines for

    - identity - file name of the given data set
    - average beat rate
    - average displacement
    - average x motion
    - average prevalence
    - average principal strain

Optionally add
    -i [idt]
where [idt] is a given identity for the data set; "output" used as
default values.

We compute, print and plot average + standard deviation for each
metric.

The output is saved as [idt].csv in 'Gather analysis data' in a
subfolder of 'Output'.

The plots are saved as [idt].png and [idt].svg in 'Gather analysis
data' in a subfolder of 'Figures'.

Åshild Telle / Simula Research Labratory / 2019

"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser

import io_funs as io

def calc_mean_std(file_list):

    N_metrics = 6
    N_files = len(file_list)

    means = np.zeros((N_files, N_metrics))
    stds = np.zeros((N_files, N_metrics))

    for i in range(N_files):
        f_in = open(file_list[i], "r")

        values = [[float(x) for x in ff.split(", ")[1:]] \
                for ff in f_in.read().split("\n")[:-1]]
        values = np.array(values)

        for j in range(5):
            means[i, j] = np.mean(values[:,j])
            stds[i, j] = np.std(values[:,j])

        f_in.close()

    return means, stds    


def plot_mean_std(means, stds, idt):
    labels = ["Beat rate", "Displacement", "X motion", "Prevalence", \
            "Principal strain", "Principal strain x direction"]
    prefixes = ["beatrate", "disp", "xmotion", "prevalence", \
            "ppstrain", "ppxstrain"]

    N = len(labels)
    N_f = len(means)

    for j in range(N):
        plt.plot(range(N_f), means[:,j], 'bo')
        plt.errorbar(range(N_f), means[:,j], yerr=stds[:,j], xerr=None, ls='none')
        plt.title(labels[j])
        #plt.xscale('log')
        plt.savefig(idt + prefixes[j] + "_iso.png")
        plt.savefig(idt + prefixes[j] + "_iso.svg")
        plt.clf() 


try:
    assert(len(sys.argv)>1)
except:
    print("Give file names as arguments; optional argument -i \
            [identity of data set].")
    exit(-1)

# get output file if given

parser = OptionParser()
parser.add_option("-i")
(options, args) = parser.parse_args()
options = vars(options)

idt = options["i"] if options["i"] is not None else "output"

de = io.get_os_delimiter()
subpath = "Output" + de + "Plot_mean_stds" + de
path = de.join((subpath + idt).split(de)[:-1])
io.make_dir_structure(path)

fout = open(subpath + idt + ".csv", "w")

output_headers = ",".join([" ", "Beat rate", "Displacement", \
        "X-motion", "Prevalence", "Principal strain", \
        "Principal strain, x-fraction"])

means, stds = calc_mean_std(args)

subpath = "Figures" + de + "Plot_mean_stds" + de
path = de.join((subpath + idt).split(de)[:-1])
io.make_dir_structure(path)
idt = subpath + de + idt

plot_mean_std(means, stds, idt)

fout.write(", ".join([str(x) for x in (["Mean: "] + list(means))]))
fout.write(", ".join([str(x) for x in (["Std: "] + list(stds))]))

fout.close()
