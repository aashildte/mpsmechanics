"""

Åshild Telle / Simula Research Laboratory / 2020
Berenice Charrez / UC Berkeley / 2020
"""

import os
import glob
from collections import defaultdict, OrderedDict
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mpsmechanics.utils.data_layer import read_prev_layer
from mpsmechanics.utils.command_line import get_input_files
from mpsmechanics.pillar_tracking.pillar_tracking import \
        track_pillars, calc_int_avg
from mpsmechanics.mechanical_analysis.mechanical_analysis import \
        analyze_mechanics

def get_info_length(f_in: str):
    """

    Extracts information about the experiment from the path,

    Args:
        f_in - path to file; information about experiment
            should be specified here

    Taguchi array examples:
L1 = length1 = average of run1 run2 run3
W2 = width2 = average of run2 run5 run8 etc.

Run#   	1	2	3	4	5	6	7	8	9
Length	1	1	1	2	2	2	3	3	3
Width	1	2	3	1	2	3	1	2	3
Exten	1	2	3	2	3	1	3	1	2
Valve	1	2	3	3	1	2	2	3	1
"""

    for run in ["Run1", "Run2", "Run3"]:
        if run in f_in:
            #print("run length1", run)
            return "Length1"

    for run in ["Run4", "Run5", "Run6"]:
        if run in f_in:
            return "Length2"

    for run in ["Run7", "Run8", "Run9"]:
        if run in f_in:
            return "Length3"

    print("Error: Was not found to have any given length")


def get_info_width(f_in: str):


    for run in ["Run1", "Run4", "Run7"]:
        if run in f_in:
            return "Width1"

    for run in ["Run2", "Run5", "Run8"]:
        if run in f_in:
            return "Width2"

    for run in ["Run3", "Run6", "Run9"]:
        if run in f_in:
            return "Width3"

    print("Error: Was not found to have any given width")

def get_info_extension(f_in: str):


    for run in ["Run1", "Run6", "Run8"]:
        if run in f_in:
            #print("run in ext1", run)
            return "Extension1"

    for run in ["Run2", "Run4", "Run9"]:
        if run in f_in:
            return "Extension2"

    for run in ["Run3", "Run5", "Run7"]:
        if run in f_in:
            return "Extension3"

    print("Error: Was not found to have any given ext")

def get_info_valve(f_in: str):

    for run in ["Run1", "Run5", "Run9"]:
        if run in f_in:
            #print("run in valve1", run)
            return "Valve1"

    for run in ["Run2", "Run6", "Run7"]:
        if run in f_in:
            return "Valve2"

    for run in ["Run3", "Run4", "Run8"]:
        if run in f_in:
            return "Valve3"

    print("Error: Was not found to have any given valve")


def read_metric_data(f_in: str, metrics: list, param_list: dict):
    """

    Gets metric information from results from the pillar tracking.

    Args:
        f_in - filename/path to BF file
        metrics - list of which metrics we're interested in

    Returns:
        dictionary with metrics as keys, corresponing calculated metric
            for the given input file as values

    """

    metric_information = {}
    #print("f_in_read:", f_in)
    track_pillars_results = read_prev_layer(f_in, track_pillars, param_list)


    for metric in metrics:
        metric_information[metric] = track_pillars_results[metric]

    return metric_information



def get_metrics_across_experiments(input_files: list,
                                   average_across: dict,
                                   param_dict: dict,
                                   metrics: list,
                                   get_info):
    """

    Extract the information we need; make it into statisitcs.

    Args:
        input_files - list of (BF) files to do analysis for
        average_across - dictionary where
            keys = which kind of thing we consider (taguchi - length, width, ext, valve)
            values = list of corresponding possible values (1, 2, 3)
        metrics - list of metrics we want to consider (force per area)

    Returns:
        list of 4 dictionaries (mean, std, num_samples, sem) where each is a
            nested dictionary with information from average_across on first level,
            metric on second level and a single value on the third level

    calc_relevant metric: gives out a dictionary with stats for given metrics
            "all_values": values,
            "folded": folded,
            "over_time_avg": over_time_avg,
            "over_time_std": over_time_std,
            "metrics_max_avg": metrics_max_avg,
            "metrics_avg_avg": metrics_avg_avg,
            "metrics_max_std": metrics_max_std,
            "metrics_avg_std": metrics_avg_std,
            "metrics_int_avg": metrics_int_avg,
            "metrics_int_std": metrics_int_std,

    """

    param_list = [{},{},{}]
    interval_list = []
    data_dict = {}

    metric_data = defaultdict(lambda: defaultdict(list))

    for f_in in input_files:
        name=os.path.splitext(f_in)
        for keys in param_dict.keys():
            name2=os.path.split(keys)
            name3, end = os.path.split(name2[0])
            if name[0] == name3:
                param_list=[{},{},{'motion_scaling_factor': param_dict[keys]}]
                #start,end = os.path.split(f_in)
                #end2 = os.path.splitext(end)
                #print("track_pillars {} -o -ms {}".format(end2[0], param_dict[keys]))
            else:
                continue

        str_information = get_info(f_in)
        metric_information = read_metric_data(f_in, metrics, param_list)        # force per area, 195x4x2
        interval_list = get_intervals(f_in)

        for arrays in metric_information:
            metrics_int_avg=calc_int_avg(metric_information[arrays], interval_list) #max force per pillar [1x #pillar]

        #print("metric int avg", metrics_int_avg)
        metric_max = get_max_metric (metrics_int_avg)  # one number = max force
        #print("metric max", metric_max)
        data_dict[f_in] = {}
        for values in data_dict:
                data_dict[f_in]["metrics_int_avg"] = metrics_int_avg
                data_dict[f_in]["metric_max"] = metric_max
                data_dict[f_in]["str_info"] = str_information

    print("data_dict", data_dict)
    metric_data = match_max(data_dict)
    #print("metric_data",metric_data)       #metric data is a dict with keys as chip_name and values as (max_force and symmetry_diff and str_info)
    stats = calc_stats (metric_data, average_across)
    return stats, metric_data

def match_max(data_dict: dict):

    metric_data = {}
    chip_name = []
    pair_dict = {}
    for (counter,f_in) in enumerate(data_dict.keys()):
        start,end = os.path.split(f_in)
        end2 = os.path.splitext(end)
        run = ["Run1", "Run2", "Run3"]
        if any (x in f_in for x in run):
            end3=end2[0].rsplit('_',3)
            chip_name.append(end3[0])
            max_top = 0
            max_bottom = 0

            for i in range (int(len(data_dict[f_in]["metrics_int_avg"])/2)):  #look at the top and bottom pillars of the same file and compare max force from top with max force from bottom
                i_2= i + int(len(data_dict[f_in]["metrics_int_avg"])/2)
                if data_dict[f_in]["metrics_int_avg"][i] > max_top:
                    max_top = data_dict[f_in]["metrics_int_avg"][i]
                if data_dict[f_in]["metrics_int_avg"][i_2] > max_bottom:
                    max_bottom = data_dict[f_in]["metrics_int_avg"][i_2]

                match_diff = (max_top-max_bottom)**2
                chip_name_counter = chip_name[counter]
                metric_data[chip_name_counter]={}
                metric_data[chip_name_counter]["max_force"] = data_dict[f_in]["metric_max"]
                metric_data[chip_name_counter]["symmetry_diff"] = match_diff
                metric_data[chip_name_counter]["str_info"] = data_dict[f_in]["str_info"]

        else:
            end3=end2[0].rsplit('_',4)
            chip_name.append(end3[0])

    pair_dict = find_duplicates(chip_name)

    for names in chip_name:     #look at same file names (top vs bottom), compare each files metric_max
        for keys in pair_dict:
            if names == keys:
                position1 = pair_dict[keys][0]
                position2 = pair_dict[keys][1]
                key_temp1 = list(data_dict)[position1]
                key_temp2 = list(data_dict)[position2]
                match_diff = (data_dict[key_temp1]["metric_max"] - data_dict[key_temp2]["metric_max"])**2

                if data_dict[key_temp1]["metric_max"] > data_dict[key_temp2]["metric_max"]:
                    metric_max = data_dict[key_temp1]["metric_max"]
                else:
                    metric_max = data_dict[key_temp2]["metric_max"]

                chip_name_pos = chip_name[position1]
                metric_data[chip_name_pos]={}
                metric_data[chip_name_pos]["max_force"] = metric_max
                metric_data[chip_name_pos]["symmetry_diff"] = match_diff
                metric_data[chip_name_pos]["str_info"] = data_dict[key_temp1]["str_info"]

    return metric_data

def find_duplicates(chip_name: list):
    new_dict = {}
    tally = defaultdict(list)
    for counter,item in enumerate(chip_name):
        tally[item].append(counter)
    for key,locs in tally.items():
        if len(locs)>1:
            new_dict[key] = locs
    return new_dict

def get_max_metric (metrics_int_avg):

    if isinstance(metrics_int_avg, float):
        metric_max = metrics_int_avg
    else:
        for values in metrics_int_avg:
            metric_max = np.nanmax(metrics_int_avg)

    return metric_max

def _walk_glob(argument_list):
    input_files = []
    for arg in argument_list:
        input_files.extend(glob.glob(arg))
    return input_files

def find_param_dict(argument_list):

    param_dict = {}
    input_files =_walk_glob([os.path.join(argument_list, "*", "*", "*")])

    for a_file in input_files:
        if a_file.find("track_pillars") != -1:
            #print("found .npy track pillar?", a_file)
            base=os.path.basename(a_file)
            scaling=base.rsplit('_',1)[1]
            scaling_factor=float(f"{scaling[0]}.{scaling[2]}")
            new_dict = {a_file : scaling_factor}
            param_dict.update(new_dict)

        else:
            continue
            #print("no folder to look for param_list")
        #print("param_dict return", param_dict)

    return param_dict

def get_intervals(f_in: str):

    interval_list = []
    analyze_mechanics_results = read_prev_layer(f_in, analyze_mechanics)
    interval_list = analyze_mechanics_results["intervals"]

    return interval_list

def calc_stats(metric_data, average_across: dict):
    """

    From a dictionary of lists to 4 dictionaries of statistical quantities.

    detects outliers

    Args:
        metric_data - nested dictionary with [combination to take average across] as
            keys on the first level, metric on the second level, and list/array
            of floats on third level

    Returns:
        mean - nested dictionary
        std - nested dictionary
        num_samples - nested dictionary
        sem - nested dictionary; where all of these have the same structure as
            metric_data, except from having a single float at last level

    """
    mean = defaultdict(lambda: defaultdict(float))
    std = defaultdict(lambda: defaultdict(float))
    num_samples = defaultdict(lambda: defaultdict(float))
    sem = defaultdict(lambda: defaultdict(float))

    for av_acr in average_across:
        for  param in average_across[av_acr]:
            temp_dict = {}
            temp_dict["max_force"] = []
            temp_dict["symmetry_diff"] = []

            for names in metric_data:
                if metric_data[names]["str_info"] == param:
                    temp_dict["max_force"].append(metric_data[names]["max_force"])
                    temp_dict["symmetry_diff"].append(metric_data[names]["symmetry_diff"])

            for keys in temp_dict:
                if len(temp_dict[keys])!= 0:
                    temp_dict[keys] = outliers(temp_dict[keys])
                    mean[param][keys] = np.mean(temp_dict[keys])
                    std[param][keys] = np.std(temp_dict[keys])
                    num_samples[param][keys] = len(temp_dict[keys])
                    sem[param][keys] = std[param][keys]/np.sqrt(num_samples[param][keys])

                else:
                    continue

    return mean, std, num_samples, sem


def outliers(array):

        #print("array:", array)
    Q1 = np.quantile(array, .25)
    Q3 = np.quantile(array, .75)
    IQR = Q3-Q1
    lowbound = Q1 - (IQR*1.5)
    upbound = Q1 + (IQR*1.5)

    new_array = []

    for data in array:
        if data >= lowbound and data <= upbound:
            new_array.append(data)
    #print("newarray:", new_array)
    return np.array(new_array)

def save_stats_to_csv(stats: list, input_files: list):
    """

    dictionary -> csv file

    TODO: Work with this to get the output format you want.

    Arguments:
        stats - list of 4 dictionaries
        output_folder - save plot here

    """

    mean, std, num_samples, sem = stats
    output_file = os.path.join(output_folder, f"force_metrics_stats.csv")

    # assume here all stats dictionaries have the same keys
    metrics_data = defaultdict(lambda: defaultdict(float))

    for str_info in mean.keys():
            for metric in mean[str_info].keys():
                metrics_data[str_info][f"{metric}_mean"] = mean[str_info][metric]
                metrics_data[str_info][f"{metric}_SEM"] = sem[str_info][metric]
                metrics_data[str_info][f"{metric}_n"] = num_samples[str_info][metric]

    dataframe = pd.DataFrame(metrics_data)
    dataframe.to_csv(output_file)
    print(f"Data saved to {output_file}.")

def save_dict_to_csv(metric_data: dict):
    """

    dictionary -> csv file

    Arguments:
        metric_data - list of a dictionary with name: max force, symmerty diff, str_info
        output_folder - save plot here
    """
    output_file = os.path.join(output_folder, f"force_metric_dict.csv")

    dataframe = pd.DataFrame(metric_data)

    dataframe.to_csv(output_file)
    print(f"Data saved to {output_file}.")

def plot_all_metrics(stats: list, output_folder: str):
    """

    This isn't as pretty as Prism plot but give a quick overview.

    Args:
        stats - list of dictionaries, expected to be [mean, std, num_samples, sem]
        metrics - metrics we want to consider
        units - corresponding units
        output_folder - save plot here

    """

    mean, _, _, sem = stats

    all_info_str = list(mean.keys())
    length=[]
    units= ["mN/mm2", "-"]
    for keys in mean:
        for count, key2 in enumerate(mean[keys].keys()):
            length.append(count)

    num_subplot = int(len(length)/len(mean.keys()))

    _, axes = plt.subplots(num_subplot, sharex=True)
    for keys in mean:
        for (axis, key2, unit) in zip(axes, mean[keys], units):
            for (index, info_str) in enumerate(all_info_str):
                axis.errorbar([index], mean[info_str][key2], yerr=sem[info_str][key2], fmt='o')

            ylabel = f"{key2} ({unit})"
            ylabel = ylabel.replace("_", " ")
            ylabel = ylabel.capitalize()
            axis.set_ylabel(ylabel)

    num_labels = len(all_info_str)
    axes[-1].set_xlim(-0.5, num_labels-0.5)    # just making it pretty
    axes[-1].set_xticks(range(num_labels))
    axes[-1].set_xticklabels(all_info_str, rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "force_metrics.png"))
    plt.show()      # comment this one out if it's annoying


if __name__ == "__main__":

    output_folder = sys.argv[1]
    input_files = get_input_files(sys.argv[1:], "BF")        # all files/folders given on the command line

    param_dict = find_param_dict (sys.argv[1])

    average_across = ["Length", "Width", "Extension", "Valve"]
    #average_across = OrderedDict({"design" : ["design1", "design2"]})
    metrics = ["force_per_area"]

    get_info_functions = [get_info_length, get_info_width, get_info_extension, get_info_valve]
    stats = [{}, {}, {}, {}]

    for avg_across, get_info in zip(average_across, get_info_functions):
        #print("function", get_info)
        av_acr = {avg_across : [f"{avg_across}{i+1}" for i in range(3)]}
        print(av_acr)
        new_stats, all_data = get_metrics_across_experiments(input_files, av_acr, param_dict, metrics, get_info)
        for i in range(4):
            stats[i] = {**stats[i], **new_stats[i]}

    output_folder = "output"                                 # you might want to change this
    os.makedirs(output_folder, exist_ok=True)

    save_dict_to_csv(all_data)
    save_stats_to_csv(stats, input_files)

    plot_all_metrics(stats, output_folder)
