# -*- coding: utf-8 -*-
"""

Metrics related functions.

Åshild Telle / Simula Research Labratory / 2019

"""

import os
import mpsmechanics as mc


def _get_plotting_properties(plot_properties, path_plots, dimensions, max_t):
    """

    Defines a dictionary which gives useful information about
    plotting properties; to be forwarded to plotting functions.

    Arguments:
        plt_id - (sorted) list of integers identifying
            which values to plot
        f_in - filename, including full path
        idt - string used for identification of data set
        dimensions - length and height of pictures used for recording
        max_t - time frame (seconds)

    Return:
        Dictionary with some useful plotting information

    """

    plt_p = {}

    # optional argument; default false
    for i in range(11):
        plt_p[int(i)] = {"plot" : False}

    for i in plot_properties:
        plt_p[i]["plot"] = True

    # get information specificly for metrics

    # other properties
    plt_p["path"] = path_plots
    plt_p["dims"] = dimensions     # size of plots over height/length
    plt_p["visual check"] = True   # extra plots if applicable
    plt_p["Tmax"] = max_t

    return plt_p


def _save_output(idt, descriptions, values, path_num):
    """

    Saves output to file.

    Arguments:
        idt    - filename
        values - dictionary of corresponding output values
        path_num - where to save given output

    """

    # interleave calc_idts, values

    output_vals = []

    headers_str = ", ".join([" "] + descriptions) + "\n"
    values_str = ", ".join([idt] + list(map(str, values))) + "\n"

    filename = os.path.join(path_num, "metrics_average.csv")
    fout = open(filename, "w")
    fout.write(headers_str)
    fout.write(values_str)
    fout.close()


def analyze_mechanics(input_files, calc_properties, plot_properties):
    """

    Calculates mechanical metrics.

    Args:
        input files - list of nd2 / csv files to perform analysis for
        calc_propeties - list of integers, which properties to calculate
        plot_properties - list of integers, which properties to plot

    """


    # default parameters

    alpha, N_d, dt, threshold_pv, threshold_mv = \
            mc.get_default_parameters()

    # TODO in parallel, based on threads? - separate function

    for f_in in input_files:
        # create directory structure
        path, idt, _ = mc.get_input_properties(f_in)
        path_num, path_plots = \
                mc.make_default_structure(path, \
                "analyze_mechanics", idt)

        print("Analyzing data set: ", idt)

        # read + preprocess data

        disp_data, scale, dimensions = mc.read_mt_file(f_in)
        disp_data = mc.do_diffusion(disp_data, alpha, N_d, over_time=True)

        # for plotting?
        T = disp_data.shape[0]
        max_t = dt*T

        plt_prop = _get_plotting_properties(plot_properties, path_plots, \
                dimensions, max_t)

        # calculations

        descriptions, values = mc.calc_metrics(disp_data, calc_properties, \
            scale, dt, plt_prop, threshold_pv, threshold_mv)

        # save as ...

        _save_output(idt, descriptions, values, path_num)

        print("Analysis of " + idt + " finished:")
        print(" * Output saved in '" + path_num + "'")

        if not plot_properties:
            print(" * Specified plots saved in '" + path_plots + "'")
