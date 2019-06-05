# -*- coding: utf-8 -*-
"""

Vector / heat map plots related functions.

Åshild Telle / Simula Research Labratory / 2019

"""

import os
import mpsmechanics as mc

def visualize_mechanics(cl_args):
    """

    TODO

    """

    input_files, calc_properties, _ = mc.get_cl_input()

    if(min(calc_properties) < 1 or max(calc_properties) > 9):
        print("Give ids in range [1, 9] as input for plotting properties.")
        exit(-1)

    # default parameters

    alpha, N_d, dt, threshold_pr, threshold_mv = \
            mc.get_default_parameters()

    # TODO in parallel, based on threads?

    for f_in in input_files:

        # create directory structure
        
        path, idt, _ = mc.get_input_properties(f_in)

        path = os.path.join(os.path.join(path, idt), "visualize_mechanics")
        mc.make_dir_structure(path)

        print("Creating plots for data set: ", idt)

        # read + preprocess data

        disp_data, scale, dimensions = mc.read_mt_file(f_in)
        disp_data = mc.do_diffusion(disp_data, alpha, N_d, over_time=True)

        over_time = False

        mc.plot_metrics2D(disp_data, dimensions, calc_properties, scale, dt, \
                threshold_pr, threshold_mv, path, over_time)

        print("Finished plotting, figures located in " + path)
