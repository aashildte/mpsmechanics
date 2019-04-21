
"""

Functions related to command line arguments

Åshild Telle / Simula Research Labratory / 2019

"""

from optparse import OptionParser


def _get_plt_properties(options, key, calc_properties):

    if(options[key] is not None):
        plt_properties = list(map(int, options[key].split(" ")))
        plt_properties.sort()

        # check subset

        for idt in plt_properties:
            if idt not in calc_properties:
                print("Error: Specified plots needs to be a subset " +\
                        "of specified metrics.")
                exit(-1)
    else:
        plt_properties = []

    return plt_properties


def get_cl_input():
    """

    Reads command line and transforms into useful variables.

    Returns:
        input_files - list of input files
        calculation identities - list of integers identifying
            values of interest (which properties to compute)
        plotting identities - list of integers identifying
            values of interest (which properties to plot)

    """
    
    # optional arguments

    parser = OptionParser()
    parser.add_option("-p")
    parser.add_option("-f")
    (options, args) = parser.parse_args()
    options = vars(options)

    try:
        assert(len(args)>1)
    except:
        print("Give files name and integers indicationg values of " +
                "interests as arguments (see the README file); " +
                " optionally '-p [indices]' and / or '-f [indices]'" +
                " to plot values as well.")
        exit(-1)

    input_files = args[:-1]

    calc_properties = list(map(int, args[-1].split(" ")))
    calc_properties.sort()

    plt_p, plt_f = [_get_plt_properties(options, key, \
                    calc_properties) for key in ("p", "f")]
    
    return input_files, calc_properties, plt_p, plt_f
