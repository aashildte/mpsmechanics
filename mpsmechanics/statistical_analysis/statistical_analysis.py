"""

Åshild Telle / Simula Research Laboratory / 2019

"""

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

from ..utils.iofuns.folder_structure import get_input_properties
from ..utils.iofuns.data_layer import read_prev_layer
from ..pillar_tracking.pillar_tracking import track_pillars
from ..mechanical_analysis.mechanical_analysis import analyze_mechanics

def _read_data(input_files, layer_name, layer_fn):
    """

    Detects folder structure; loads data.

    TODO maybe move to iofuns.
    TODO 2 - for now this won't work on Windows

    """

    all_maxima = defaultdict(lambda : defaultdict(dict))

    for f_in in input_files:
        path, filename, _ = get_input_properties(f_in)

        dose, media = path.split("/")[-2:]

        data = read_prev_layer(f_in, layer_name, layer_fn)
        print(data["over_time_avg"].keys())
        all_maxima[dose][media][filename] = \
                np.max(data["over_time_avg"]["displacement_um"])  # e.g.

    doses_keys = list(all_maxima.keys())
    media_keys = list(all_maxima[doses_keys[0]].keys())

    return doses_keys, media_keys, all_maxima


def _find_correct_layer(layer):
    """

    The script intends to find data/call functions to calculate
    data if needed - layers needs to be in specific set of functions
    as defined specifically. For now this includes mechanical analysis
    and pillar tracking. With some adaptions we could possibly
    also include things like calcum and action potential traces.

    """

    fn_map = {"track_pillars" : track_pillars,
            "analyze_mechanics" : analyze_mechanics}

    assert layer in fn_map.keys(), \
            "Error: No corresponding function found"

    return layer, fn_map[layer]


def _sort_keys(keys, sort_by):
    """

    Given a set of keys, sort it by substrings as indicated in
    sort_by. Default here *can* be alphabetically maybe, but for
    now we'll raise an error.

    TODO need to check that no key in sort_by is a substring of
    another one.

    """
    
    assert len(keys) == len(sort_by), \
            "Error: Length not preserved? Check substrings."

    keys_sorted = []

    for key1 in sort_by:
        for key2 in keys:
            if key1 in key2:
                keys_sorted.append(key2)
                break

    assert len(keys) == len(keys_sorted), \
            "Error: Length not preserved? Check substrings."

    return keys_sorted



def calculate_stats_chips(input_files, layers, sort_by):
    """

    For now this is only meant to give average and std/avg for
    different doses, maximum over time, average over all pillars.
    Eventually we can do more here ...

    """
    
    layers = layers.split(" ")
    sort_by = sort_by.split(" ")

    for layer in layers:
        layer_name, layer_fn = _find_correct_layer(layer)

        doses_keys, media_keys, all_maxima = _read_data(input_files, \
                layer_name, layer_fn)

        # expected structure: doses / media / data

        stats = defaultdict(lambda : defaultdict(tuple))

        doses_keys = _sort_keys(doses_keys, sort_by)

        avgs = []
        stds = []

        for dose in doses_keys:
            for media in media_keys:
                values = np.array(list(all_maxima[dose][media].values()))
                avg = np.mean(values)
                std = np.std(values)/avg        # normalised!
                stats[dose][media] = (avg, std)
                # TODO to file or to dictionary -> next layer
                avgs.append(avg)
                stds.append(std)
                print(dose, media, avg, std)
