# -*- coding: utf-8 -*-

"""

Functions related to command line arguments

Åshild Telle / Simula Research Labratory / 2019

"""

import os
import glob
import numpy as np
import mps

def _valid_input_file(input_file):
    
    if not "BF" in input_file:
        return False
    
    if not "nd2" in input_file:
        return False

    return True


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

    # read in files

    input_args = []
    for x in s_files:
        input_args.extend(glob.glob(x))
    
    # walk through all files and folders
    input_files = []

    for x in input_args:
        if os.path.isfile(x):
            if _valid_input_file(x):
                input_files.append(x)
        elif os.path.isdir(x):
            # if folders, replace with all files in subfolders
            
            for root, _, files in os.walk(x):
                for f in files:
                    filename = os.path.join(root, f)
                    if _valid_input_file(filename):
                        input_files.append(filename)

    return input_files

