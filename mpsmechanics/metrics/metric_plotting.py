"""

File connected to metrics.py. These functions gather information
useful for plotting and output descriptions.

TODO - lots of state variables; make class instead?

Åshild Telle / Simula Research Labratory / 2019

"""

import sys
import numpy as np

from . import preprocessing as pp
from . import operations as op
from . import angular as an
from . import heartbeat as hb
from . import mechanical_properties as mc

    
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

    return ["Beat rate",
            "Prevalence",
            "Displacement",
            "Displacement - x projection",
            "Displacement - y projection",
            "Velocity",
            "Velocity - x projection",
            "Velocity - y projection",
            "Principal strain",
            "Principal strain - x projection",
            "Principal strain - y projection"]

def get_pr_types():
    """
    Array of all property identities
    """
    pr_types = ["beat_rate",
                "prevalence",
                "displacement",
                "displacement_x",
                "displacement_y",
                "velocity",
                "velocity_x",
                "velocity_y",
                "prstrain",
                "prstrain_x",
                "prstrain_y"]

    return pr_types

def get_scales():
    """
    For over time plots.
    """

    scales = [None]*11

    return scales

def get_pr_id(pr_type):
    """

    From a string to an integer. Just a 1-1 thing but might be useful
    to avoid mistakes based on different properties.

    TODO create dictionary instead

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

    yscales = get_scales()

    for i in range(8):
        if i in ppl.keys():
            ppl[i]["idt"] = idt + "_" + suffixes[i]
            ppl[i]["title"] = titles[i]
            ppl[i]["yscale"] = yscales[i]
            ppl[i]["Tmax"] = Tmax


if __name__ == "__main__":

    print("TODO - unit tests?")
