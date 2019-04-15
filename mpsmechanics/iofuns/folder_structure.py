"""

Åshild Telle / Simula Research Labratory / 2019

"""

import sys
import os


def get_path(filename):
    """

    Remove relative paths + file suffix

    """

    # strip f_in for all relative paths

    while(".." in filename):
        r_ind = filename.find("..") + 3
        filename = filename[r_ind:]

    # and for file type / suffix

    r_ind = filename.find(".")
    filename = filename[:r_ind]

    return filename


def get_idt(filename):
    """

    Strip path + file suffix

    """

    filename = os.path.normpath(filename).split(os.path.sep)[-1]
    return filename.split(".")[0]


def make_dir_structure(path):
    """

    Makes a directory structure based on a given path.

    """

    dirs = os.path.normpath(path).split(os.path.sep)
    
    acc_d = ""

    for d in dirs:
        acc_d = os.path.join(acc_d, d)
        if not (os.path.exists(acc_d)):
            os.mkdir(acc_d)


