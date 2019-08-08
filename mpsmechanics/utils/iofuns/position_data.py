# -*- coding: utf-8 -*-

"""

Function for reading initial position file

Åshild Telle / Simula Research Labratory / 2019

"""

import numpy as np
import pandas as pd


def read_pt_file(f_in):
    """

    Reads in values for pillar coordinates + radii.

    Args:
        f_in - Filename

    Returns;
        Numpy array of dimensions P x 3, P being the number of
            points; entries being x, y, radius for all points.

    """

    if(".csv" in f_in):
        return _read_pt_file_csv(f_in)
    elif(".npy" in f_in):
        return _read_pt_file_nd2(f_in)
    else:
        print("Error: Uknown file formate.")

def _read_pt_file_nd2(f_in):
    """

    Reads in values for pillar coordinates + radii.

    Args:
        f_in - Filename

    Returns;
        Numpy array of dimensions P x 3, P being the number of
            points; entries being x, y, radius for all points.

    """
    data = np.load(f_in, allow_pickle=True).item()

    # convention : longitudal = x; transverse = y
    df = pd.DataFrame(data=data)
    x_pos = df["positions_longitudinal"].values
    y_pos = df["positions_transverse"].values
    radii = df["radii"].values

    return np.swapaxes(np.array((x_pos, y_pos, radii)), 0, 1)


def _read_pt_file_csv(f_in):
    """

    Reads in values for pillar coordinates + radii.

    Args:
        f_in - Filename

    Returns;
        Numpy array of dimensions P x 3, P being the number of
            points; entries being x, y, radius for all points.

    """

    f = open(f_in, "r")

    lines = f.read().split("\n")[1:]

    if not lines[-1]:
        lines = lines[:-1]      # ignore last line if empty

    p_values = [[int(x) for x in line.split(",")] \
                        for line in lines]

    # flip x and y; different conventions for these two scripts

    for i in range(len(p_values)):
        x, y, r = p_values[i]
        p_values[i] = [y, x, r]

    p_values = np.array(p_values)

    f.close()

    return p_values
