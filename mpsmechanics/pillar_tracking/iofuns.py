# -*- coding: utf-8 -*-
"""

IO functions related to track_pillar scripts


Åshild Telle / Simula Research Labratory / 2019
"""

import os

from ..iofuns import command_line as cl
from ..iofuns import writetofile as wf
from ..iofuns import folder_structure as fs


def pts_write_all_values(all_values, mpoints, path):
    """

    Output to files: T x N values for each pillar

    Args:
        all_values - numpy array of dimension T x P x N x 2
        coords - midpoints; numpy array of dimension P x 2
        path - save here

    """

    P = len(mpoints)

    assert all_values.shape[1] == P, "Dimension mismatch?"

    for p in range(P):
        coords = mpoints[p]
        f_suffix = (
            "_".join(
                ["pillar", str(int(coords[0])), str(int(coords[1]))]
            )
            + ".csv"
        )

        filename = os.path.join(path, f_suffix)
        wf.write_position_values(all_values[:, p], filename)


def pts_write_max_values(
    mid_values, max_indices, coords, path, prop
):
    """

    Writes values at maximum displacement to a file.

    Args:
        mid_values - T x P x 2 numpy array
        max_indices - list-alike structure for indices of maxima
        coords - coordinates of midpoints
        path - save file here

    """

    assert len(coords) == mid_values.shape[1], "dimension mismatch?"

    filename = os.path.join(path, prop + "_at_maxima.csv")

    output_d = {}

    for p in range(len(coords)):
        key = str(coords[p, 0]) + " " + str(coords[p, 1])
        output_d[key] = []

        for m in max_indices:
            output_d[key].append(mid_values[m, p])

    wf.write_max_values(max_indices, output_d, filename)


def define_paths(f_disp, out_dir="track_pillars"):
    """
    Define and create paths for output of track_pillar script.

    Folder structure:
    out_dir
        numerical_output
            positions_all_time_step
            displacement_maxima
            force_maxima
        figures
            positions_all_time_step
            displacement_maxima
            force_maxima

    Args:
        f_disp - string input with file name, used to define path
        out_dir - directory that contains all the computed data
            (plots, .csv files, ...)
    
    Returns:
        Dictionary with path structure; keys being "num_all", "num_max", \
            "plt_all", "plt_max"
        idt - string with last part of file name
    """

    path, idt, _ = fs.get_input_properties(f_disp)

    path_num, path_plots = fs.make_default_structure(
        path, out_dir, idt
    )

    paths = []

    for p in [path_num, path_plots]:
        for a in [
            "positions_all_time_step",
            "displacement_maxima",
            "force_maxima",
        ]:
            pt = os.path.join(p, a)
            fs.make_dir_structure(pt)
            paths.append(pt)

    paths_dir = {}
    paths_dir["num_all"] = paths[0]
    paths_dir["num_max"] = paths[1]
    paths_dir["plt_all"] = paths[2]
    paths_dir["plt_max"] = paths[3]

    return paths_dir, idt
