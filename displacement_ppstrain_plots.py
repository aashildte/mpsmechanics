
import sys
import numpy as np
import matplotlib.colors as cl
import threading as th

import preprocess_data as pp
import mechanical_properties as mc
import plot_vector_field as pl
import read_data as io

def find_values(f_in, disp, strain):
    disp_data = io.read_disp_file(f_in)
    strain_data = mc.compute_principal_strain(disp_data)

    time_step = pp.get_max_ind(pp.get_overall_movement(disp_data))

    disp.append(disp_data[time_step])
    strain.append(strain_data[time_step])

    print("Success 1!")

def get_range(data):
    return (np.min(data), np.max(data))


try:
    assert(len(sys.argv)>1)
except:
    print("Give file names + output file as arguments.")
    exit(-1)

f_out = sys.argv[-1]

alpha = 0.75
N_d = 5

idts = []
filenames = []

for f_in in sys.argv[1:]:
    last_fn = f_in.split("/")[-1].split(".")

    # check suffix - if not a csv file, skip this one

    try:
        prefix, suffix = last_fn
        assert(suffix == "csv")
    except:
        continue

    idts.append(prefix)
    filenames.append(f_in)


# first parallel loop

threads = []
disp_all = []
strain_all = []
threads = []

for f_in in filenames:

    t = th.Thread(target = find_values, \
            args = (f_in, disp_all, strain_all))
    t.start()

    threads.append(t)

# lock

for t in threads:
    t.join()

disp_all = np.asarray(disp_all)
strain_all = np.asarray(strain_all)

# prepare plotting

minmax1 = get_range(disp_all)
minmax2 = get_range(strain_all)

print("Ranges: ", minmax1, minmax2)

norm1 = cl.Normalize(minmax1[0], minmax1[1])
norm2 = cl.LogNorm(1, minmax2[1] - minmax2[0])  # positive values only

norms = [norm1, norm2]

dimensions = (664.30, 381.55)
labels = ["Displacement", "Principal strain"]

N = len(disp_all)

filter1 = lambda x : x.copy()

def filter2(D):
    X, Y = D.shape[:2]

    ND = np.zeros(D.shape)

    for x in range(X):
        for y in range(Y):
           if(abs(D[x, y, 1] - 1) > 1E-10):
               ND[x, y] = D[x, y]

    return ND

filters = [filter1, filter2]

# second parallel loop

for i in range(N):
    vector_fields = [disp_all[i], strain_all[i]]
    t = th.Thread(target=pl.plot_direction_and_magnitude, \
            args=(vector_fields, norms, labels, filters, dimensions, idts[i]))
    t.run()
