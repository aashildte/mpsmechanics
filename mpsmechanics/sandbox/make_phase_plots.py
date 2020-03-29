
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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

    plt.plot(time_equal[0], avg_per_beat)
    plt.savefig("strain.png")
    plt.clf()

    time_avg = time_equal[0] - time_equal[0][0]

    return time_avg, avg_per_beat, data["time"], strain_avg, pacing


def get_fl_avg(input_file):
    mps_ob = mps.MPS(input_file)
    
    data = mps.analysis.analyze_mps_func(mps_ob)
    
    trace_all = data["unchopped_data"]["trace"]
    time_all = data["unchopped_data"]["time"]
    pacing = data["unchopped_data"]["pacing"]

    avg_per_beat = data["chopped_data"]["trace_1std"]
    time_avg = data["chopped_data"]["time_1std"]

    avg_per_beat /= max(avg_per_beat)

    return time_avg, avg_per_beat, time_all, trace_all, pacing

def make_plots_over_time(fig, outer_axis, strain_values, ca_values, ap_values):
    inner = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer_axis)

    axes = []

    for (i, values) in enumerate((strain_values, ca_values, ap_values)):
        time = values[2]
        trace = values[3]
        pacing = values[4]
        pacing = (max(trace)/5)*pacing

        axis = plt.Subplot(fig, inner[i, 0])
        axis.plot(time, trace)
        axis.plot(time, pacing)
        axes.append(axis)
        fig.add_subplot(axis)

    axes[0].set_ylabel("Strain")
    axes[0].set_xlabel("Time")
    axes[1].set_ylabel("Ca")
    axes[1].set_xlabel("Time")
    axes[2].set_ylabel("AP")
    axes[2].set_xlabel("Time")


def make_phase_plots(fig, outer_axis, strain_values, ca_values, ap_values):
    strain_fun, ca_fun, ap_fun = \
            [interp1d(values[0], values[1], fill_value="extrapolate") \
                for values in (strain_values, ca_values, ap_values)]

    time = np.linspace(0, 1000, 1000)
    
    strain = strain_fun(time)
    ca = ca_fun(time)
    ap = ap_fun(time)

    inner = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer_axis)

    axis = plt.Subplot(fig, inner[0, 0])
    axis.plot(time, ap, time, ca, time, strain)
    axis.legend(["AP", "Ca", "Strain"])
    axis.set_xlabel("Time")
    axis.set_ylabel("Ca / AP / Strain")
    fig.add_subplot(axis)

    axis = plt.Subplot(fig, inner[0, 1])
    axis.plot(ca, strain)
    axis.set_xlabel("Calcium")
    axis.set_ylabel("Strain")
    fig.add_subplot(axis)

    axis = plt.Subplot(fig, inner[1, 0])
    axis.plot(ap, strain)
    axis.set_xlabel("AP")
    axis.set_ylabel("Strain")
    fig.add_subplot(axis)
    
    axis = plt.Subplot(fig, inner[1, 1])
    axis.plot(ca, ap)
    axis.set_xlabel("Ca")
    axis.set_ylabel("AP")
    fig.add_subplot(axis)


def make_strain_ca_ap_plots(BF_file, Cyan_file, Red_file):

    strain_values = get_strain_avg(BF_file)
    ca_values = get_fl_avg(Cyan_file)
    ap_values = get_fl_avg(Red_file)
    
    fig = plt.figure(figsize=(20, 10))
    outer = gridspec.GridSpec(1, 2)

    make_plots_over_time(fig, outer[0], strain_values, ca_values, ap_values)
    make_phase_plots(fig, outer[1], strain_values, ca_values, ap_values)

    plt.savefig("strain_ca_ap.png")

BF_file = sys.argv[1]
Cyan_file = sys.argv[2]
Red_file = sys.argv[3]

assert "BF" in BF_file, \
        f"Error: Expected BF file as first argument, not {BF_file}."
assert "Cyan" in BF_file, \
        f"Error: Expected Cyan file as first argument, not {Cyan_file}."
assert "Red" in BF_file, \
        f"Error: Expected Red file as first argument, not {Red_file}."

make_strain_ca_ap_plots(BF_file, Cyan_file, Red_file)
