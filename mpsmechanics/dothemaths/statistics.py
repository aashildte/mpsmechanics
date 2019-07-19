"""

Åshild Telle / Simula Research Laboratory / 2019

"""

import numpy as np

from .heartbeat import calc_beat_minmax

def calc_for_each_key(init_data, fn):
    """

    Performs a given operations for a dictionary with similar data
    sets.

    Args:
        init_data - dictionary with attributes as keys, T x R values,
            where T is an integer; R an integer or tuple of integers
        fn - function f : R -> S

    Returns:
        A numpy array with resulting values, of dimension T x R

    """

    d = {}

    for key in init_data.keys():
        d[key] = fn(init_data[key])

    return d


def chip_statistics(data, displacement, dt):
    """

    Args:
        data - dictionary with attributes as keys, data over space
            and time as values; each array (value) is assumed to
            be of dimension T x A x B x D where
                A and B describes point distribution
                D gives dimension of values

    Returns;
        dictionary with values, folded distributions, moments, and
            some values of general interest

    """
    fn_folded = lambda x: np.linalg.norm(x, axis=3)
    fn_mean = lambda x: np.mean(x, axis=(1, 2))
    fn_std = lambda x: np.std(x, axis=(1, 2))

    d_all = {}

    d_all["all_values"] = data
    d_all["folded"] = calc_for_each_key(data, fn_folded)
    d_all["over_time_avg"] = calc_for_each_key(d_all["folded"], fn_mean)
    d_all["over_time_std"] = calc_for_each_key(d_all["folded"], fn_std)

    # general variables

    d_all["time"] = np.linspace(0, dt*len(displacement), len(displacement))
    d_all["minima"], d_all["maxima"] = calc_beat_minmax(displacement)

    return d_all
