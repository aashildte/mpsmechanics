# -*- coding: utf-8 -*-

"""

Functions related to command line arguments

Åshild Telle / Simula Research Labratory / 2019

"""

import os
import glob


def get_input_files(s_files):
    """
    
    Reads in files and foldersfrom a list (like sys.argv), filters
    out those that are likely to be valid files.

    Assesses whether the script should be run in debug mode or not.

    Args:
        list of files, folders, etc., e.g. from command line
    
    Returns:
        debug - boolean value; perform in debug mode or not
        input_files - list, BF nd2 files from s_files

    """

    debug = "-d" in s_files or "--debug" in s_files

    # read in files

    input_args = []
    for x in s_files:
        input_args.extend(glob.glob(x))
    
    # condition: only include BF/nd2 files
    cond = lambda x : "BF" in x and "nd2" in x

    # walk through all files and folders
    input_files = []

    for x in input_args:
        if os.path.isfile(x):
            if cond(x):
                input_files.append(x)
        elif os.path.isdir(x):
            # if folders, replace with all files in subfolders
            
            for root, _, files in os.walk(x):
                for f in files:
                    filename = os.path.join(root, f)
                    if cond(f):
                        input_files.append(filename)

    return debug, input_files

