
import sys
import numpy as np
import matplotlib.pyplot as plt

import mps
import mpsmechanics as mc

from scipy.interpolate import interp1d


def get_disp_avg(BF_file):
    mps_ob = mps.MPS(BF_file)
    data = mc.read_prev_layer(BF_file, mc.analyze_mechanics, [], False)
    avg = data["over_time_avg"]["displacement"]

    chopped = mps.analysis.chop_data_with_pacing(avg, \
            mps_ob.time_stamps, mps_ob.pacing)

    min_len = min([len(chopped.data[i]) for i in range(len(chopped.data))])

    chopped_equal = [d[:min_len] for d in chopped.data]
    time_equal = [t[:min_len] for t in chopped.times]

    avg = np.mean(chopped_equal, axis=0)
    time = time_equal[0]

    return avg, time

def get_ca_avg(Cyan_file):
    mps_ob = mps.MPS(Cyan_file)
    
    data = mps.analysis.analyze_mps_func(mps_ob)

    avg = data["chopped_data"]["trace_1std"]
    time = data["chopped_data"]["time_1std"]

    return avg, time

BF_file = sys.argv[1]
Cyan_file = sys.argv[2]

disp_avg, disp_time = get_disp_avg(BF_file)
ca_avg, ca_time = get_ca_avg(Cyan_file)

disp_fun = interp1d(disp_time, disp_avg, fill_value="extrapolate") 
ca_fun = interp1d(ca_time, ca_avg, fill_value="extrapolate") 

times = np.linspace(0, 1000, 1000)
disp_vals = disp_fun(times)
ca_vals = ca_fun(times)

plt.plot(disp_vals, ca_vals)
plt.xlabel("Displacement")
plt.ylabel("Calcium")
plt.savefig("phase_plot.png")
plt.show()
