
"""

Given a list of input files on the command line (typically all
files in a given directory), where each file contains two values,
this computes the maximum of all the first and all the second
values over all files.

Run as
    python calc_maxima.py [files]

The maximum values are printed to the terminal.

Corresponding color bars (normal and lognormal resp.) are also
plotted and saved in a folder called Figures/calc_maxima
(Figures\calc_maxima for Windows).

The idea is to use this combined with find_range.py, which give
a set of files of maximum values as output, and plot_disp_strain,
which plots displacement and strain plots scaled with output values
of this script.

"""

import sys
import matplotlib.colors as cl
import matplotlib.pyplot as plt
import matplotlib as mpl

import io_funs as io

# get data

max1 = []
max2 = []

for filename in sys.argv[1:]:
    fin = open(filename, "r")

    values = list(map(float, fin.read().split(", ")[:-1]))

    max1.append(values[0])
    max2.append(values[1])

m1, m2 = max(max1), max(max2)
print(m1, m2)

# plotting

path = "Figures/calc_maxima"

io.make_dir_structure(path)

de = io.get_os_del()

fig, ax = plt.subplots(figsize=(6, 1))
fig.subplots_adjust(bottom=0.5)

norm1 = mpl.colors.Normalize(0.0, m1)
norm2 = mpl.colors.LogNorm(0.01, m2)

cb1 = mpl.colorbar.ColorbarBase(ax,
                                norm=norm1,
                                orientation='horizontal')
cb1.set_label('Displacement')

plt.savefig(path + de + "Displacement.png", dpi=1000)
#plt.savefig(path + de + "Displacement.svg")

plt.clf()

cb1 = mpl.colorbar.ColorbarBase(ax,
                                norm=norm2,
                                orientation='horizontal')
cb1.set_label('Principal strain')

plt.savefig(path + de + "Strain.png", dpi=1000)
#plt.savefig(path + de + "Strain.svg")
