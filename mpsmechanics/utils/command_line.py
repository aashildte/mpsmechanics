# -*- coding: utf-8 -*-

"""

Functions related to command line arguments

Åshild Telle / Simula Research Labratory / 2019

"""

import os
import glob


def _valid_input_file(input_file, filetype):

    if not filetype in input_file:
        return False

    if not ".nd2" in input_file:
        return False

    return True


def _walk_glob(argument_list):

    input_files = []
    for arg in argument_list:
        input_files.extend(glob.glob(arg))

    return input_files


def get_input_files(argument_list, filetype="BF"):
    """

    Reads in files and folders from a list (like sys.argv), filters
    out those that are likely to be valid files.

    Assesses whether the script should be run in debug mode or not.

    Args:
        s_files - list of files, folders, etc., e.g. from command line
        t - type; default BF, can be Cyan (or ?)

    Returns:
        debug - boolean value; perform in debug mode or not
        input_files - list, BF nd2 files from s_files

    """
    input_files = []

    for a_file in _walk_glob(argument_list):
        if os.path.isfile(a_file):
            if _valid_input_file(a_file, filetype):
                input_files.append(a_file)
        elif os.path.isdir(a_file):
            # if folders, replace with all files in subfolders
            for root, _, files in os.walk(a_file):
                for f_in in files:
                    filename = os.path.join(root, f_in)
                    if _valid_input_file(filename, filetype):
                        input_files.append(filename)

    return input_files


def add_default_parser_arguments(parser, file_type):
    """

    Adds arguments which absolutely all scripts need.

    Args:
        parser - argument parser
        file_type - "BF", "Cyan" or "Red"

    """

    parser.add_argument(
        "input_files",
        help=f"{file_type} files to run the script for",
        nargs="+",
    )

    parser.add_argument(
        "-o",
        "--overwrite",
        help="Recalculate and overwrite previous data/plots, for specific script.",
        action="store_true",
    )

    parser.add_argument(
        "-oa",
        "--overwrite_all",
        help="Recalculate and overwrite previous data/plots, for all layers/dependencies.",
        action="store_true",
    )

    parser.add_argument(
        "-d",
        "--debug_mode",
        help="Run script in debug mode",
        action="store_true",
    )


def add_parameters_parser_arguments(parser, level):
    """

    Adds arguments which are needed on lower levels; where
        track_motion = level 0
        analyze_mechanics = level 1
    such that all scripts dependent on these will get the same
    parameter set.

    Args:
        parser - argument parser
        level - 0, 1 or higher

    Returns:
        list of strings which identify the different
            arguments added to the argument parser

    """

    all_keys = []

    if level >= 0:
        l0_keys = []
    if level >= 1:
        l1_keys = []

    return [l0_keys, l1_keys]


def add_pillar_tracking_arguments(parser):
    """

    Adds arguments needed for pillar tracking scripts.

    Args:
        parser - argument parser

    Returns:
        list of strings which identify the different
            arguments added to the argument parser

    """
    
    parser.add_argument(
        "-ms",
        "--motion_scaling_factor",
        default=1,
        help="Scaling factor for motion; 1 = original.",
        type=float,
    )

    return ["motion_scaling_factor"]


def add_animation_parser_arguments(parser):
    """

    Adds arguments needed for animation scripts.

    Args:
        parser - argument parser

    Returns:
        list of strings which identify the different
            arguments added to the argument parser

    """

    parser.add_argument(
        "-a",
        "--animate",
        help="Make animations or just peak plots.",
        action="store_true",
    )

    parser.add_argument(
        "-s",
        "--scaling_factor",
        default=1,
        help="Scaling factor for fps; 1 = real time, 0.5 half speed",
        type=float,
    )

    parser.add_argument(
        "-t",
        "--time_step",
        default=None,  # = "peak"
        help="Which time step to plot heat maps for; default is at peak.",
        type=int,
    )
    
    parser.add_argument(
        "-rmax",
        "--range_max",
        default=None,
        help="Which range to use for plots (color bar); gives upper bound.",
        type=float,
    )

    parser.add_argument(
        "-rmin",
        "--range_min",
        default=None,
        help="Which range to use for plots (color bar); gives upper bound.",
        type=float,
    )


    return ["animate", "scaling_factor", "time_step", "range_max", "range_min"]

def add_focus_parser_arguments(parser):
    """

    Adds arguments needed for mesh over movie scripts.

    Args:
        parser - argument parser

    """

    parser.add_argument(
        "-w",
        "--width",
        default=100,
        help="Width (in pixels) for focus area.",
        type=int,
    )

    parser.add_argument(
        "-x",
        "--xcoord",
        default=100,
        help="Middle point; x direction (along chamber), in pixels.",
        type=int,
    )

    parser.add_argument(
        "-y",
        "--ycoord",
        default=100,
        help="Middle point; y direction (across chamber), in pixels.",
        type=int,
    )

    parser.add_argument(
        "-st",
        "--step",
        default=1,
        help="Plot every [step] line in mesh.",
        type=int,
    )

    return ["width", "xcoord", "ycoord", "step"]


def add_subdivision_arguments(parser):
    """

    Adds arguments needed to chose selected metrics.

    Args:
        parser - argument parser

    """

    parser.add_argument(
        "-sx",
        "--sub_xdir",
        default=5,
        help="Number of subdomains along the chamber.",
        type=int,
    )

    parser.add_argument(
        "-sy",
        "--sub_ydir",
        default=1,
        help="Number of subdomains across the chamber.",
        type=int,
    )

    return ["sub_xdir", "sub_ydir"]


def add_metrics_arguments(parser):
    """

    Adds arguments needed to chose selected metrics.

    Args:
        parser - argument parser

    """

    parser.add_argument(
        "-m",
        "--metrics",
        default="displacement",
        help="Which metrics to plot/make figures for (one string, separate multiple ones by space).",
        type=str,
    )

    return ["metrics"]

def split_parameter_dictionary(vargs, keys):
    """
    # TODO there might be a more elegant way to do this:

    Args:
        vargs - flat dictionary, all keys
        keys - keys for different levels, 2D list

    Retunrs:
        list of dictionaries where each sublist
            in keys corresponds to the keys in each
            dictionary
    """

    parameters = []
    for key_list in keys:
        kwargs = {}
        for key in key_list:
            kwargs[key] = vargs.pop(key)
        parameters.append(kwargs)

    return parameters
