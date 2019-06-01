"""

Åshild Telle / Simula Research Labratory / 2019

"""

import sys
import os


def get_input_properties(filename):
    """

    Splits file into three parts: Path, filename, extension

    Arguments:
        filename - full path including filename

    Returns:
        path - path to filename
        filename - filename in that directory
        ext - file extension

    """

    path, tail = os.path.split(filename)
    filename, ext = tail.split(".")

    return path, filename, ext

def make_default_structure(path, subfolder, idt):
    """

    Define/create structure for
        output path -> subfolder -> idt
    and two subfolders, "numerical output" and "figures"
    in the last one.

    Returns:
        path to numerical output
        path to plots

    """

    f_path = os.path.join(\
        os.path.join(path, idt), \
        subfolder)

    path_num = os.path.join(f_path, "numerical_output")
    path_plots = os.path.join(f_path, "figures")

    make_dir_structure(path_num)
    make_dir_structure(path_plots)

    return path_num, path_plots


def make_dir_structure(path):
    """

    Makes a directory structure based on a given path; for every
    directory specified it's created unless it already exists.

    Arguments:
        path, final "/" not needed

    """

    dirs = os.path.normpath(path).split(os.path.sep)
    
    if not os.path.exists(path):
        os.makedirs(path)
