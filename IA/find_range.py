"""

Finds the value range of displacement and strain for a given data set.

This file is meant to be combined with plot_disp_strain.py and
find_maximum.py which; together these three files give a combined
direction + magnitude plot for both displacement and principal 
strain. As we want all output values on the same scale we need to 
find the maximum over all data sets; as the files are large, it is 
too memory- and timeconsuming to do this calculation and the
resulting plots combined, and is best done for each file separately.

Run this file as

    python3 find_range.py [input file]

to get the maximum magnitude of displacement and strain; this is
saved in a folder disp_strain_range with the file name given
as input being used as an identity.

"""

import os
import sys
import numpy as np
import matplotlib.colors as cl

import dothemaths.operations as op
import dothemaths.preprocessing as pp
import dothemaths.mechanical_properties as mc
import dothemaths.plot_vector_field as pl
import iofuns.io_funs as io

def find_max_values(f_in):
    """

    Given the displacement data, this function calculates the maximum
    magnitude of displacement and principal strain, as an l2-norm.

    Arguments:
        f_in - filename

    Returns:
        maximum displacement
        maximum principal strain

    """
    
    xlen = 664.30

    disp_data, scale = io.read_file_csv(f_in, xlen)
    
    time_step = op.calc_max_ind(op.calc_norm_over_time(disp_data))
    disp_t = disp_data[time_step]

    strain_t = mc.calc_principal_strain(disp_t, over_time=False)

    return np.max(scale*disp_t), np.max(strain_t)

try:
    assert(len(sys.argv)>1)
except:
    print("Give file name as first positional argument.")
    exit(-1)


f_in = sys.argv[1]
idt = os.path.split(f_in)[-1].split(".")[0]

max1, max2 = find_max_values(f_in)

# save values

path = os.path.join("Output", "Find_range")
io.make_dir_structure(path)
fout = open(os.path.join(path, "range_" + idt + ".csv"), "w")
fout.write(str(max1) + ", " + str(max2) + ", ")
fout.close()
