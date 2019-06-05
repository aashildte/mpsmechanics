# -*- coding: utf-8 -*-

"""

Functions related to command line arguments

Åshild Telle / Simula Research Labratory / 2019

"""

from argparse import ArgumentParser


def _string_to_ints(calc_str):
    """

    Converts a string on the form "2 4 6" to a list [2, 4, 6].

    Args:
        calc_str: String with integers separated by spaces

    Returns:
        Sorted list containing the same integers.

    """
    if not calc_str:
        return []

    calc_str = list(map(int, calc_str.split(" ")))
    calc_str.sort()

    return calc_str


def get_cl_input(arg_keys=()):
    """

    Reads command line and transforms into useful variables.

    Args:
        arg_keys: Optional, key pairs for options for which
            to add to argument parser
    Returns:
        argument parser structure

    """

    parser = ArgumentParser()

    # default arguments

    parser.add_argument("vars", nargs="+")

    # optional arguments
    for option in arg_keys:
        parser.add_argument(*option[0], **option[1])

    args = parser.parse_args()

    input_files = args.vars[:-1]
    calc_properties = args.vars[-1]

    # per default, "p" or "plot" is reserved for ids for all scripts
    try:
        args.plot = _string_to_ints(args.plot)
    except:
        pass

    try:
        assert input_files
        calc_properties = _string_to_ints(calc_properties)

    except:
        print("Give files name and integers indicationg values of " +
              "interests + possibly optional arguments on the " +
              "command line (see README file and/or top of " +
              "given script).")
        exit(-1)

    return input_files, calc_properties, args
