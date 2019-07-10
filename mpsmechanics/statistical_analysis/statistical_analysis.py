"""

Åshild Telle / Simula Research Laboratory / 2019

"""

from collections import defaultdict

from ..utils.iofuns.folder_structure import get_input_properties
from ..pillar_tracking.pillar_tracking import track_pillars

def average(idt, descriptions, values, path_num):
    """

    TODO review

    Saves output to file.

    Arguments:
        idt    - filename
        values - dictionary of corresponding output values
        path_num - where to save given output

    """

    # interleave calc_idts, values

    output_vals = []

    headers_str = ", ".join([" "] + descriptions) + "\n"
    values_str = ", ".join([idt] + list(map(str, values))) + "\n"

    filename = os.path.join(path_num, "metrics_average.csv")
    fout = open(filename, "w")
    fout.write(headers_str)
    fout.write(values_str)
    fout.close()

def _read_data(input_files):
    """

    Detects folder structure; loads data.

    TODO maybe move to iofuns.
    TODO 2 - for now this won't work on Windows

    """

    all_maxima = defaultdict(dict)

    for f_in in input_files:
        path, filename, _ = get_input_properties(f_in)

        dose = path.split("/")[-2]

        data = read_prev_layer(f_in, "track_pillars", track_pillars)

        all_maxima[dose][filename] = np.max(data["displacement_um"])  # e.g.

    return all_maxima


def calculate_stats_chips(input_files):
    """

    For now this is only meant to give average and std/avg for different doses,
    maximum over time, average over all pillars. Eventually we can do more here ...

    """

    all_maxima = _read_data(input_files)

    # TODO we need a logical way to sort the keys!!! sometimes it's dose1, dose2, .. and
    # sometimes 0nM, 1nM, 10nM, ... and sometimes 1nM, 10nM, 100nM, 1uM, 10uM

    stats = {}

    for key in all_maxima.keys():
        avg = np.mean(all_maxima[key].values())
        std = np.std(all_maxima[key].values())/avg
        stats[key] = (avg, std)

    print("avg, std/avg for each dose: ")
    print(stats)
