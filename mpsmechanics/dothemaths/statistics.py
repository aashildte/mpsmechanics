"""

Åshild Telle / Simula Research Laboratory / 2019

"""

import numpy as np

from .heartbeat import calc_beat_maxima_time, calc_beat_intervals


def find_nonzero_scale(displacement):

    dims = displacement.shape

    t_dim, x_dim, y_dim, n_dim = dims
    
    entries = np.sum(np.any((displacement != 0), axis=0))
    
    return x_dim*y_dim/entries



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
    d_all = {}
    

    # some transformations
    fn_folded = lambda x: np.linalg.norm(x, axis=3)
    fn_mean = lambda x: np.mean(x, axis=(1, 2))
    fn_std = lambda x: np.std(x, axis=(1, 2))
    fn_max = lambda x : max(x)


    d_all["all_values"] = data
    d_all["folded"] = calc_for_each_key(data, fn_folded)
    d_all["over_time_avg"] = calc_for_each_key(d_all["folded"], fn_mean)
    d_all["over_time_std"] = calc_for_each_key(d_all["folded"], fn_std)
    
    # general variables

    d_all["time"] = np.linspace(0, dt*len(displacement), len(displacement))
    d_all["maxima"] = calc_beat_maxima_time(d_all["over_time_avg"]["displacement"])
    d_all["intervals"] = calc_beat_intervals(d_all["over_time_avg"]["displacement"])
    
    fn_meanmax = lambda x : np.mean([max(x[i1:i2]) \
            for (i1, i2) in d_all["intervals"]])

    # metrics

    d_all["metrics_max"] = calc_for_each_key(d_all["over_time_avg"], fn_max)
    d_all["metrics_mean"] = calc_for_each_key(d_all["over_time_avg"], fn_meanmax)

    # separate for beatrate

    d_all["metrics_max"]["beatrate"] = np.max(np.diff(d_all["maxima"]))
    d_all["metrics_mean"]["beatrate"] = np.mean(np.diff(d_all["maxima"]))

    return d_all
