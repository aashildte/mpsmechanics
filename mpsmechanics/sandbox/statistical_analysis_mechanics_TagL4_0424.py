"""

Åshild Telle / Simula Research Laboratory / 2020
Berenice Charrez / UC Berkeley / 2020

"""

import os
from collections import defaultdict, OrderedDict
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mpsmechanics.utils.data_layer import read_prev_layer
from mpsmechanics.utils.command_line import get_input_files
from mpsmechanics.mechanical_analysis.mechanical_analysis import \
        analyze_mechanics


def get_info_ec(f_in: str):
    """

    Extracts information about the experiment from the path,

    Args:
        f_in - path to file; information about experiment
            should be specified here

    Taguchi array examples:
pluronic1 = + ; pluronic2 = -
Fib1 = fibrinogen conc = 15ng/ul; Fib2 = 100ng/ul
EC:SC1 = 10:90 ; EC:SC1 = 90:10

Run#   	                1	2	3	4
EC:SC and pluronic  	1	2	2	1
Fib 	                1	2	1	2
"""

    for run in ["Run1", "Run4"]:
        if run in f_in:
            #print("run length1", run)
            return "ECSC1"

    for run in ["Run2", "Run3"]:
        if run in f_in:
            return "ECSC2"

    print("Error: Was not found to have any given length")


def get_info_fib(f_in: str):


    for run in ["Run1", "Run3"]:
        if run in f_in:
            #print("run width1", run)
            return "Fib1"

    for run in ["Run2", "Run4"]:
        if run in f_in:
            return "Fib2"

    print("Error: Was not found to have any given width")


def read_metric_data(f_in: str, metrics: list, type_metric: str):
    """

    Gets metric information from results from the mechanical analysis.

    Args:
        f_in - filename/path to BF file
        metrics - list of which metrics we're interested in
        type_metric - str giving kind of metrics; expected to be in
            ["metrics_max_avg", "metrics_avg_avg", "metrics_max_std", "metrics_avg_std", "metrics_int_avg", "metrics_int_std""]

    Returns:
        dictionary with metrics as keys, corresponing calculated metric
            for the given input file as values

    """

    exp_type_metrics = ["metrics_max_avg", "metrics_avg_avg", "metrics_max_std", "metrics_avg_std", "metrics_int_avg", "metrics_int_std"]
    assert type_metric in exp_type_metrics, \
            f"Error: type_metric expected to be in {exp_type_metrics}, " + \
                    f"but argument {type_metric} given."

    metric_information = {}
    #print("f_in_read:", f_in)
    analyze_mechanics_results = read_prev_layer(f_in, analyze_mechanics)


    all_metrics = analyze_mechanics_results[type_metric]

    #print(all_metrics)
    for metric in metrics:
        metric_information[metric] = all_metrics[metric]

    return metric_information



def get_metrics_across_experiments(input_files: list,
                                   average_across: dict,
                                   metrics: list,
                                   type_metric: str,
                                   get_info):
    """

    Extract the information we need; make it into statisitcs.

    Args:
        input_files - list of (BF) files to do analysis for
        average_across - dictionary where
            keys = which kind of thing we consider (e.g. "doses")
            values = list of corresponding possible values (e.g. "dose0", "dose1")
        metrics - list of metrics we want to consider
        type_metric - str giving kind of metrics; expected to be in
            ["metrics_max_avg", "metrics_avg_avg", "metrics_max_std", "metrics_max_std"]

    Returns:
        list of 4 dictionaries (mean, std, num_samples, sem) where each is a
            nested dictionary with information from average_across on first level,
            metric on second level and a single value on the third level

    """

    # first dimension: str of combination of across values; second dimension: metric

    metric_data = defaultdict(lambda: defaultdict(list))
    all_data={}

    for f_in in input_files:
        start,end = os.path.split(f_in)
        end2 = os.path.splitext(end)
        name=end2[0]
        all_data[name]={}
        str_information = get_info(f_in)         # e.g. "dose1_1Hz"
        metric_information = read_metric_data(f_in, metrics, type_metric)        # e.g. {"velocity" : 2}

        for keys in metric_information:
            all_data[name][keys] = metric_information[keys]

        for metric in metrics:
            metric_val = metric_information[metric]
            if not np.isnan(metric_val):
                metric_data[str_information][metric].append(metric_val)

    stats = calc_stats(metric_data)

    return stats, all_data


def calc_stats(metric_data):
    """

    From a dictionary of lists to 4 dictionaries of statistical quantities.

    TODO: Nothing detects outliers - here's where you probably would include that.

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

    for str_info in metric_data.keys():
        for metric in metric_data[str_info].keys():

            metric_data[str_info][metric] = outliers(metric_data[str_info][metric])
            mean[str_info][metric] = np.mean(metric_data[str_info][metric])
            std[str_info][metric] = np.std(metric_data[str_info][metric])
            num_samples[str_info][metric] = len(metric_data[str_info][metric])
            sem[str_info][metric] = std[str_info][metric]/np.sqrt(num_samples[str_info][metric])

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

def save_dict_to_csv(metric_data: dict):
    """

    dictionary -> csv file

    Arguments:
        metric_data - list of a dictionary with name: max force, symmerty diff, str_info
        output_folder - save plot here
    """
    print(metric_data)

    output_file = os.path.join(output_folder, f"metric_dict.csv")

    dataframe = pd.DataFrame(metric_data)

    dataframe.to_csv(output_file)
    print(f"Data saved to {output_file}.")


def save_data_to_csv(stats: list, input_files: list):
    """

    dictionary -> csv file

    TODO: Work with this to get the output format you want.

    Arguments:
        stats - list of 4 dictionaries
        output_folder - save plot here

    """

    mean, std, num_samples, sem = stats
    output_file = os.path.join(output_folder, f"metrics_stats.csv")

    # assume here all stats dictionaries have the same keys
    metrics_data = defaultdict(lambda: defaultdict(float))


    for str_info in mean.keys():
            for metric in mean[str_info].keys():
                #print(str_info)
                #print(metric)
                metrics_data[str_info][f"{metric}_mean"] = mean[str_info][metric]
                metrics_data[str_info][f"{metric}_SEM"] = sem[str_info][metric]
                metrics_data[str_info][f"{metric}_n"] = num_samples[str_info][metric]


    metrics_data["input_files"]["all files"] = input_files
    metrics_data["values"]["all values"] = mean

#    for str_info in mean.keys():
#        for metric in mean[str_info].keys():

#            metrics_data[f"{str_info}_{metric}"]["mean"] = mean[str_info][metric]
#            metrics_data[f"{str_info}_{metric}"]["sem"] = sem[str_info][metric]
#            metrics_data[f"{str_info}_{metric}"]["n"] = num_samples[str_info][metric]


    dataframe = pd.DataFrame(metrics_data)
    dataframe.to_csv(output_file)
    print(f"Data saved to {output_file}.")


def plot_all_metrics(stats: list, metrics: list, units: list, output_folder: str):
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

    _, axes = plt.subplots(len(metrics), sharex=True)

    for (metric, unit, axis) in zip(metrics, units, axes):
        for (index, info_str) in enumerate(all_info_str):
            axis.errorbar([index], mean[info_str][metric], yerr=sem[info_str][metric], fmt='o')

        ylabel = f"{metric} ({unit})"
        ylabel = ylabel.replace("_", " ")
        ylabel = ylabel.capitalize()

        axis.set_ylabel(ylabel)

    num_labels = len(all_info_str)
    axes[-1].set_xlim(-0.5, num_labels-0.5)    # just making it pretty
    axes[-1].set_xticks(range(num_labels))
    axes[-1].set_xticklabels(all_info_str, rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "metrics.png"))
    plt.show()      # comment this one out if it's annoying

def get_mean(input_files: list, metrics: list, type_metric: str):
    """
    Extract the information we need; give us mean values.

    Args:
        input_files - list of (BF) files to do analysis for
        metrics - list of metrics we want to consider
        type_metric - "metrics_avg_avg"

    Returns:
        list of 1 dictionaries (mean)
    """

    metric_data = defaultdict(lambda: defaultdict(list))
    mean = defaultdict(lambda: defaultdict(float))

    for f_in in input_files:

        #str_information = get_info(f_in)         # e.g. "dose1_1Hz"
        metric_information = read_metric_data(f_in, metrics, type_metric)        # e.g. {"velocity" : 2}

        for metric in metrics:
            metric_val = metric_information[metric]
            if not np.isnan(metric_val):
                metric_data[f_in][metric].append(metric_val)

    for str_info in metric_data.keys():
        for metric in metric_data[str_info].keys():

            metric_data[str_info][metric] = outliers(metric_data[str_info][metric])
            mean[str_info][metric] = np.mean(metric_data[str_info][metric])

    return mean

if __name__ == "__main__":

    output_folder = sys.argv[1]
    input_files = get_input_files(sys.argv[1:], "BF")        # all files/folders given on the command line

    average_across = ["ECSC", "Fib"]
    #average_across = OrderedDict({"design" : ["design1", "design2"]})
    metrics = ["xmotion","principal_strain", "tensile_strain", "compressive_strain"]

    get_info_functions = [get_info_ec, get_info_fib]
    stats = [{}, {}, {}, {}]

    for avg_across, get_info in zip(average_across, get_info_functions):
        #print("function", get_info)
        av_acr = {avg_across : [f"{avg_across}{i+1}" for i in range(2)]}
        print(av_acr)
        new_stats, all_data = get_metrics_across_experiments(input_files, av_acr, metrics, "metrics_int_avg", get_info)
        for i in range(4):
            stats[i] = {**stats[i], **new_stats[i]}

    output_folder = "output"                                 # you might want to change this
    os.makedirs(output_folder, exist_ok=True)

    save_dict_to_csv(all_data)
    save_data_to_csv(stats, input_files)

    units = ["-","-", "-", "-"]               # you *can* get these from the analyze_mechanics but probably just easier
                                        # to give them manually. Keep updated with metrics.

    plot_all_metrics(stats, metrics, units, output_folder)

"""
    metrics2 = ["displacement", "displacement_maximum_difference", "principal_strain"]      #doesnt have disp max diff in the script!
    mean = {}
    mean = get_mean(input_files, metrics2, "metrics_avg_avg")

    print("MAXIMUM")
    for metric in metrics2:
        max_key = ""
        max_val = 0
        max_delta = 0
        max_delta_key = ""


        for f_in in mean.keys():
            this_val = mean[f_in][metric]
            if this_val > max_val:
                max_val = this_val
                max_key = f_in

            delta = (mean[f_in]["displacement_maximum_difference"] - mean[f_in]["displacement"]) / mean[f_in]["displacement"]
            if delta > max_delta:
                max_delta = delta
                max_delta_key = f_in

        print(metric, max_key)

    print("max (delta dispMaxDiff & disp) file", max_delta_key)

    print("MINIMUM")
    for metric in metrics2:
        min_key = ""
        min_val = 10
        min_delta = 10
        min_delta_key = ""

        for f_in in mean.keys():
            this_val = mean[f_in][metric]
            if this_val < min_val:
                min_val = this_val
                min_key = f_in

            delta = mean[f_in]["displacement_maximum_difference"] - mean[f_in]["displacement"] / mean[f_in]["displacement"]
            if delta < min_delta:
                min_delta = delta
                min_delta_key = f_in

        print(metric, min_key)

    print("min (delta dispMaxDiff & disp) file", min_delta_key)

"""
