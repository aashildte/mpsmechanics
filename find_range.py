
import sys
import numpy as np
import matplotlib.colors as cl

import preprocessing as pp
import mechanical_properties as mc
import plot_vector_field as pl
import io_funs as io

def find_minmax_values(f_in):
    disp_data = io.read_disp_file(f_in)
    strain_data = mc.compute_principal_strain(disp_data)

    time_step = pp.get_max_ind(pp.get_overall_movement(disp_data))

    disp_m = pp.calculate_magnitude(np.array([disp_data[time_step]]))
    strain_m = pp.calculate_magnitude(np.array([strain_data[time_step]]))

    return np.max(disp_m), np.max(strain_m)

try:
    assert(len(sys.argv)>1)
except:
    print("Give file name as first positional argument.")
    exit(-1)

f_in = sys.argv[1]
idt = f_in.split("/")[-1].split(".")[0]

max1, max2 = find_minmax_values(f_in)

# prepare plotting

fout = open("nobackup/range_" + idt + ".csv", "w")
fout.write(str(max1) + ", " + str(max2) + ", ")
fout.close()
