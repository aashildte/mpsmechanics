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

    if not "nd2" in input_file:
        return False

    return True


def _walk_glob(argument_list):

    input_files = []
    for arg in argument_list:
        input_files.extend(glob.glob(arg))

    return input_files


def get_input_files(argument_list, filetype="BF"):
    """

    Reads in files and foldersfrom a list (like sys.argv), filters
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
    parser.add_argument("input_files", \
            help=f"{file_type} files to run the script for", \
            nargs="+")
    
    parser.add_argument("-o", "--overwrite", \
            help="Recalculate and overwrite previous data/plots..",
            action="store_true")


    parser.add_argument("-d", "--debug_mode", \
            help="Run script in debug mode",
            action="store_true")


def add_animation_parser_arguments(parser, default_scaling_factor):
    parser.add_argument("-a", "--animate", \
            help="Make animations or just peak plots.",
            action="store_true")

    parser.add_argument("-s", "--scaling_factor", \
            default=default_scaling_factor,
            help="Scaling factor for fps; 1 = real time, 0.5 half speed",
            type=float)


def add_parameters_parser_arguments(parser):
    parser.add_argument("-f", "--filter_strain", \
            default=0,
            type=int)


