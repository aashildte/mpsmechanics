"""

Given output metrics, this script gathers and evaluates these.

Run script as (with python 3)

    python plot_mean_stds_drug_studies.py [drug doses] [drug name] folder1 folder2 ... 

where folder1, folder2, etc are output values of chips for the
different doses, i.e. containing csv files consisting of

    data set identity
    property identity
    property value
    property identity
    property value

    ...

Optionally add
    -i [idt]
where [idt] is a given identity for the data set; "output" used as a
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
import os
import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser
from collections import defaultdict

import metric_plotting as mp
import io_funs as io

def create_folder_structure(idt):
    de = io.get_os_delimiter()
    subpath = "Output" + de + "Plot_mean_stds" + de
    path = de.join((subpath + idt).split(de)[:-1])
    io.make_dir_structure(path)

    fout = open(subpath + idt + ".csv", "w")

    subpath = "Figures" + de + "Plot_mean_stds" + de
    path = de.join((subpath + idt).split(de)[:-1])
    io.make_dir_structure(path)
    idt = subpath + de + idt


def calc_mean_std_folders(folders):
    means = defaultdict(list)
    stds = defaultdict(list)
    de = io.get_os_delimiter()

    for folder in folders:

        input_files = []
        for root, dirs, files in os.walk(folder):
            for f in files:
                if f.endswith(".csv"):
                    input_files.append(folder + de + f)

        idts, ms, ss = calc_mean_std(input_files)

        for j in range(len(idts)):
            i = idts[j]
            m = ms[j]
            s = ss[j]

            means[i].append(m)
            stds[i].append(s)

    idts = [x for x in means.keys()]
    idts.sort()
    
    return idts, means, stds


def calc_mean_std(file_list):

    N_files = len(file_list)

    values = defaultdict(list)

    for i in range(N_files):
        f_in = open(file_list[i], "r")
        lines = f_in.read()
        f_values = [float(x) for x in lines.split(", ")[1:]]

        for i in range(int(len(f_values)/2)):
            idt = int(f_values[2*i])
            val = f_values[2*i + 1]

            values[idt].append(val)

        f_in.close()

    idts = []
    means = []
    stds = []

    for i in values.keys():
        val = np.array(values[i])
        m = np.mean(val)
        s = np.std(val)

        idts.append(i)
        means.append(m)
        stds.append(s)

    return idts, means, stds    


def plot_mean_std(means, stds, doses, title):

    print(means)
    print(stds)
    print(doses)
    print(title)
    print("TODO: Plot these values")
    """ 
    for j in range(N):
        plt.plot(range(N_f), means[:,j], 'bo')
        plt.errorbar(range(N_f), means[:,j], yerr=stds[:,j], xerr=None, ls='none')
        plt.title(labels[j])
        #plt.xscale('log')
        plt.savefig(idt + prefixes[j] + "_iso.png")
        plt.savefig(idt + prefixes[j] + "_iso.svg")
        plt.clf() 
    """

try:
    doses = sys.argv[1]
    drug = sys.argv[2]
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
fout = open(idt, "w")

# folder structure

create_folder_structure(idt)

# read in files, do calculations

titles = mp.get_pr_headers()

idts, means, stds = calc_mean_std_folders(args[2:])

# output
for (i, m, s) in zip(idts, means, stds):
    title = titles[i] + " (" + drug + ")"
    plot_mean_std(m, s, doses, title)

print("TODO: Implement output values")

fout.close()
