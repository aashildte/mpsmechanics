"""

Åshild Telle / Simula Research Laboratory / 2020

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


def get_file_info(f_in: str, average_across: dict):
    """

    Extracts information about the experiment from the path,
    using pattern matching.

    Args:
        f_in - path to file; information about experiment
            should be specified here
        average_across - dictionary where
            keys give what kind of difference we are looking at (e.g. doces, pacing)
            values give corresponding possible values

    """

    list_information = []

    for key in average_across.keys():
        for corr_value in average_across[key]:
            if corr_value in f_in:
                list_information += [str(corr_value)]
                break

    return "_".join(list_information)


def read_metric_data(f_in: str, metrics: list, type_metric: str):
    """

    Gets metric information from results from the mechanical analysis.

    Args:
        f_in - filename/path to BF file
        metrics - list of which metrics we're interested in
        type_metric - str giving kind of metrics; expected to be in
            ["metrics_max_avg", "metrics_avg_avg", "metrics_max_std", "metrics_max_std"]

    Returns:
        dictionary with metrics as keys, corresponing calculated metric
            for the given input file as values

    """

    exp_type_metrics = ["metrics_max_avg", "metrics_avg_avg", "metrics_max_std", "metrics_max_std"]
    assert type_metric in exp_type_metrics, \
            f"Error: type_metric expected to be in {exp_type_metrics}, " + \
                    f"but argument {type_metric} given."

    metric_information = {}
    analyze_mechanics_results = read_prev_layer(f_in, analyze_mechanics)

    all_metrics = analyze_mechanics_results[type_metric]

    for metric in metrics:
        metric_information[metric] = all_metrics[metric]

    return metric_information



def get_metrics_across_experiments(input_files: list,
                                   average_across: dict,
                                   metrics: list,
                                   type_metric: str):
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

    for f_in in input_files:
        str_information = get_file_info(f_in, average_across)           # e.g. "dose1_1Hz"
        metric_information = read_metric_data(f_in, metrics, type_metric)        # e.g. {"velocity" : 2}

        for metric in metrics:
            metric_data[str_information][metric].append(metric_information[metric])

    stats = calc_stats(metric_data)

    return stats


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
            mean[str_info][metric] = np.mean(metric_data[str_info][metric])
            std[str_info][metric] = np.std(metric_data[str_info][metric])
            num_samples[str_info][metric] = len(metric_data[str_info][metric])
            sem[str_info][metric] = std[str_info][metric]/np.sqrt(num_samples[str_info][metric])

    return mean, std, num_samples, sem


def save_data_to_csv(stats: list):
    """

    dictionary -> csv file

    TODO: Work with this to get the output format you want.

    Arguments:
        stats - list of 4 dictionaries
        output_folder - save plot here

    """

    mean, std, num_samples, sem = stats
    
    output_file = os.path.join(output_folder, "metrics_stats.csv")

    # assume here all stats dictionaries have the same keys
    metrics_data = defaultdict(lambda: defaultdict(float))

    for str_info in mean.keys():
        for metric in mean[str_info].keys():

            metrics_data[f"{str_info}_{metric}"]["mean"] = mean[str_info][metric]
            metrics_data[f"{str_info}_{metric}"]["sem"] = sem[str_info][metric]
            metrics_data[f"{str_info}_{metric}"]["n"] = num_samples[str_info][metric]


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
    axes[-1].set_xticklabels(all_info_str)
 
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "metrics.png"))
    plt.show()      # comment this one out if it's annoying


if __name__ == "__main__":

    input_files = get_input_files(sys.argv[1:], "BF")        # all files/folders given on the command line 

    average_across = OrderedDict({"design" : ["design1", "design2"]})
    metrics = ["velocity", "principal_strain"]

    stats = get_metrics_across_experiments(input_files, average_across, metrics, "metrics_avg_avg")

    output_folder = "output"                                 # you might want to change this
    os.makedirs(output_folder, exist_ok=True)

    save_data_to_csv(stats)

    units = ["um/s", "-"]               # you *can* get these from the analyze_mechanics but probably just easier
                                        # to give them manually. Keep updated with metrics.

    plot_all_metrics(stats, metrics, units, output_folder)
