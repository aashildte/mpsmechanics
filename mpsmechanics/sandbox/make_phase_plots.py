
import sys
import numpy as np
import matplotlib.pyplot as plt

import mps
import mpsmechanics as mc

from scipy.interpolate import interp1d


def get_strain_avg(BF_file):
    mps_ob = mps.MPS(BF_file)
    data = mc.read_prev_layer(BF_file, mc.analyze_mechanics, [], False)
    strain_avg = data["over_time_avg"]["principal_strain"]

    pacing = mps_ob.pacing.copy()
    
    chopped = mps.analysis.chop_data_with_pacing(strain_avg, \
            mps_ob.time_stamps, pacing)

    min_len = min([len(chopped.data[i]) for i in range(len(chopped.data))])

    chopped_equal = [_d[:min_len] for _d in chopped.data]
    time_equal = [_t[:min_len] for _t in chopped.times]
 
    avg_per_beat = np.mean(chopped_equal, axis=0)
    avg_per_beat /= max(avg_per_beat)

    return avg_per_beat, time_equal[0]


def get_fl_avg(input_file):
    mps_ob = mps.MPS(input_file)
    
    data = mps.analysis.analyze_mps_func(mps_ob)
    
    avg_per_beat = data["chopped_data"]["trace_1std"]
    time = data["chopped_data"]["time_1std"]

    avg_per_beat /= max(avg_per_beat)

    return avg_per_beat, time


def make_strain_ca_ap_plots(BF_file, Cyan_file, Red_file):
    
    strain_avg, strain_time = get_strain_avg(BF_file)
    ca_avg, ca_time = get_fl_avg(Cyan_file)
    ap_avg, ap_time = get_fl_avg(Red_file)

    strain_fun = interp1d(strain_time, strain_avg, fill_value="extrapolate") 
    ca_fun = interp1d(ca_time, ca_avg, fill_value="extrapolate") 
    ap_fun = interp1d(ap_time, ap_avg, fill_value="extrapolate") 

    time = np.linspace(0, 1000, 1000)
    
    strain = strain_fun(time)
    ca = ca_fun(time)
    ap = ap_fun(time)

    _, axes = plt.subplots(2, 2, figsize=(10, 10))

    axes = axes.flatten()

    axes[0].plot(ca, strain)
    axes[1].plot(ap, strain)
    axes[2].plot(ca, ap)
    axes[3].plot(time, ap, time, ca, time, strain)
    axes[3].legend(["AP", "Ca", "Strain"])

    axes[0].set_xlabel("Calcium")
    axes[0].set_ylabel("Strain")

    axes[1].set_xlabel("AP")
    axes[1].set_ylabel("Strain")

    axes[2].set_xlabel("Ca")
    axes[2].set_ylabel("AP")

    axes[3].set_xlabel("Time")
    axes[3].set_ylabel("Ca / AP / Strain")

    plt.tight_layout()
    #plt.savefig("phase_plot.png")
    plt.show()


BF_file = sys.argv[1]
Cyan_file = sys.argv[2]
Red_file = sys.argv[3]

make_strain_ca_ap_plots(BF_file, Cyan_file, Red_file)
