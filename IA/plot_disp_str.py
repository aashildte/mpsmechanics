
import os
import sys
import numpy as np
import matplotlib.colors as cl

import dothemaths.operations as op
import dothemaths.preprocessing as pp
import dothemaths.mechanical_properties as mc
import dothemaths.plot_vector_field as pl
import iofuns.io_funs as io

def read_values(f_in):

    xlen = 664.30

    disp_data, scale = io.read_file_csv(f_in, xlen)
    
    time_step = op.calc_max_ind(op.calc_norm_over_time(disp_data))
    disp_t = disp_data[time_step]

    strain_t = mc.calc_principal_strain(disp_t, over_time=False)

    return scale*disp_t, strain_t


def get_range(data):
    return (np.min(data), np.max(data))

try:
    assert(len(sys.argv)>1)
except:
    print("Give file names + max1, max2 as arguments.")
    exit(-1)

max1, max2 = map(float, sys.argv[2:])

path = os.path.join("Figures", "Plot disp str")
io.make_dir_structure(path)

alpha = 0.75
N_d = 5

f_in = sys.argv[1]

last_fn = f_in.split("/")[-1].split(".")
prefix, suffix = last_fn
 
disp, strain = read_values(f_in)
norm1 = cl.Normalize(0.0, max1)
norm2 = cl.LogNorm(.1, max2 + .1)  # positive values only
#norm2 = cl.Normalize(0.0, max2)

norms = [norm1, norm2]

dimensions = (664.30, 381.55)
labels = ["Displacement", "Principal strain"]

# filter values by 0 displacement

X, Y = disp.shape[:2]
for x in range(X):
    for y in range(Y):
        if (np.linalg.norm(disp[x, y]) < 1E-10):
            strain[x, y] = np.array([1E-14, 1E-14])

vector_fields = [disp, strain]

pl.plot_direction_and_magnitude(vector_fields, norms, labels, dimensions, path, prefix)
