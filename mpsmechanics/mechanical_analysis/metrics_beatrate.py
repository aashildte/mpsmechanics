"""

Åshild Telle / Simula Research Laboratory / 2019

"""

import numpy as np

from ..dothemaths.heartbeat import calc_beatrate
from ..dothemaths.operations import calc_magnitude


def calc_beatrate_metric(disp_data, time, maxima, intervals):
    """

    Calculates beatrate + derived statistical quantities.

    Args:
        disp_data - displacement data, T x X x Y x 2 numpy array
        time - corresponding time slots
        maxima - indices of peaks
        intervals - indices giving interval subdivision

    Return:
        dictionary with key "beatrate", value another
            dictionary with given standardised output

    """
    disp_folded = calc_magnitude(disp_data)
    beatrate_spatial = \
            calc_beatrate(disp_folded, maxima, intervals, time)

    over_time_avg = np.mean(beatrate_spatial, axis=(1, 2))
    over_time_std = np.std(beatrate_spatial, axis=(1, 2))

    if len(over_time_avg) > 1:
        metrics_max_avg = np.max(over_time_avg)
        metrics_avg_avg = np.mean(over_time_avg)
        metrics_max_std = np.max(over_time_std)
        metrics_avg_std = np.mean(over_time_std)
    else:
        metrics_max_avg = np.nan
        metrics_avg_avg = np.nan
        metrics_max_std = np.nan
        metrics_avg_std = np.nan

    info = {"all_values" : beatrate_spatial,
            "folded" : beatrate_spatial,
            "over_time_avg" : over_time_avg,
            "over_time_std" : over_time_std,
            "metrics_max_avg" : metrics_max_avg,
            "metrics_avg_avg" : metrics_avg_avg,
            "metrix_max_std" : metrics_max_std,
            "metrix_avg_std" : metrics_avg_std,
            "unit" : r"$\mu m / s$",
            "range" : (0, np.nan),
            "range_folded" : (0, np.nan)}

    return {"beatrate" : info}
