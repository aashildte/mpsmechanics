"""

File connected to metrics.py. These functions gather information
useful for plotting and output descriptions.

Åshild Telle / Simula Research Labratory / 2019

"""

import sys
import numpy as np

import io_funs as io
import preprocessing as pp
import operations as op
import angular as an
import heart_beat as hb
import mechanical_properties as mc

    
def get_default_parameters(p_id, test_file, T):
    """

    Default parameters for unit tests across different files.

    Arguments:
        p_id - properties identity (usually displacement)
        test_file - identity for file we run unit tests for
        T - time frame

    Returns:
        Dictionary determining some properties for plotting.

    """
    de = io.get_os_delimiter()

    ppl = {}
    ppl[p_id] = {}

    ppl[p_id]["plot"] = True
    ppl[p_id]["Tmax"] = T
    ppl[p_id]["title"] = "Displacement"
    ppl[p_id]["idt"] = "unit_tests" + test_file
    ppl[p_id]["yscale"] = None

    ppl["visual check"] = True
    ppl["dims"] = (1, 1)
    ppl["idt"] = "test"
    ppl["path"] = "Figures" + de + "Unit tests"
    ppl["visual check"] = False

    io.make_dir_structure(ppl["path"])

    return ppl

def get_pr_headers():
    """
    Array of all titles.
    """

    return ["Beat rate", "Displacement", "X-motion", "Y-motion",
            "Prevalence", "Principal strain",
            "Principal strain - x projection", "Principal strain - y projection"]

def get_pr_types():
    """
    Array of all property identities
    """
    pr_types = ["beat_rate", "displacement", "xmotion", "ymotion",
            "prevalence", "prstrain", "xprstrain", "yprstrain"]

    return pr_types

def get_pr_id(pr_type):
    """

    From a string to an integer. Just a 1-1 thing but might be useful
    to avoid mistakes based on different properties.

    Argument:
        pr_type - possible ones given in pr_types

    Returns:
        integer coupled to given property

    """
    pr_types = get_pr_types()

    for i in range(len(pr_types)):
        if(pr_type == pr_types[i]):
            return i

    print("Error: Could not identify property of interest.")
    exit(-1)


def add_plt_information(ppl, idt, Tmax):
    """
    Save some general information about the metrics calculated in
    this file. These are used for visual output (plots) for
    filenames, labels, etc.

    Arguments:
        ppl - dictionary which will be altered
        idt - string identity of given data set
        Tmax - time frame for experiment

    """


    suffixes = get_pr_types()
    titles = get_pr_headers()

    yscales = [None, None, None, None, (0, 1), None, None, None]

    for i in range(8):
        if i in ppl.keys():
            ppl[i]["idt"] = idt + "_" + suffixes[i]
            ppl[i]["title"] = titles[i]
            ppl[i]["yscale"] = yscales[i]
            ppl[i]["Tmax"] = Tmax


if __name__ == "__main__":

    print("TODO - unit tests?")
