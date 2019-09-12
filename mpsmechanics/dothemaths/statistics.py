"""

Åshild Telle / Simula Research Laboratory / 2019

"""

import numpy as np

from mpsmechanics.dothemaths.heartbeat import \
        calc_beat_maxima, calc_beat_intervals


def calc_filter_time(dist):
    return np.any((dist != 0), axis=-1)

def calc_filter_all(dist):
    return np.broadcast_to(np.any((dist != 0), axis=(0,-1)), dist.shape[:3])
 

def calc_for_each_key(init_data, fn, filter_map):
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
        d[key] = fn(init_data[key], filter_map[key])

    return d


def fn_mean(x, filter_x):
    y = np.zeros(x.shape[0])

    for t in range(x.shape[0]):
        z = np.extract(filter_x[t], x[t])
        y[t] = np.mean(z)

    return y


def fn_std(x, filter_x):
    y = np.zeros(x.shape[0])

    for t in range(x.shape[0]):
        z = np.extract(filter_x[t], x[t])
        y[t] = np.std(z)

    return y


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
    
    filter_all = calc_filter_all(data["displacement"])
    filter_time = calc_filter_time(data["displacement"])
    
    #filter_all = np.tile("True", dist.shape[:3])     # original values, no filter applied

    time_filter = {"displacement" : filter_all, "velocity" : filter_all, \
            "prevalence" : filter_all, "angle" : filter_time, "xmotion": filter_time, \
            "principal strain": filter_all}

    # some transformations
    fn_folded = lambda x, _: np.linalg.norm(x, axis=3)
    fn_max = lambda x, _ : max(x)
    
    d_all["all_values"] = data
    d_all["folded"] = calc_for_each_key(data, fn_folded, time_filter)

    d_all["over_time_avg"] = calc_for_each_key(d_all["folded"], fn_mean, time_filter)
    d_all["over_time_std"] = calc_for_each_key(d_all["folded"], fn_std, time_filter)
     
    # general variables
    #d_all["time"] = np.linspace(0, (1/dt)*len(displacement), len(displacement))
    d_all["maxima"] = calc_beat_maxima(d_all["over_time_avg"]["displacement"])
    d_all["intervals"] = calc_beat_intervals(d_all["over_time_avg"]["displacement"])

    fn_meanmax = lambda x, _ : np.mean([max(x[i1:i2]) \
            for (i1, i2) in d_all["intervals"]])

    # metrics

    d_all["metrics_max_avg"] = calc_for_each_key(d_all["over_time_avg"], fn_max, time_filter)
    d_all["metrics_avg_avg"] = calc_for_each_key(d_all["over_time_avg"], fn_meanmax, time_filter)
    #d_all["metrics_std_avg"] = calc_for_each_key(d_all["over_time_avg"], fn_std, time_filter)
    d_all["metrics_max_std"] = calc_for_each_key(d_all["over_time_std"], fn_max, time_filter)
    d_all["metrics_avg_std"] = calc_for_each_key(d_all["over_time_std"], fn_meanmax, time_filter)
    #d_all["metrics_std_std"] = calc_for_each_key(d_all["over_time_std"], fn_std, time_filter)

    # separate for beatrate

    if len(d_all["maxima"]) > 1:
        d_all["metrics_max_avg"]["beatrate"] = 1/dt*np.max(np.diff(d_all["maxima"]))
        d_all["metrics_avg_avg"]["beatrate"] = 1/dt*np.mean(np.diff(d_all["maxima"]))
        #d_all["metrics_std_avg"]["beatrate"] = np.std(1/dt*np.diff(d_all["maxima"]))
        d_all["metrics_max_std"]["beatrate"] = 0
        d_all["metrics_avg_std"]["beatrate"] = 0
        #d_all["metrics_std_std"]["beatrate"] = 0
    else:
        d_all["metrics_max_avg"]["beatrate"] = 0
        d_all["metrics_avg_avg"]["beatrate"] = 0
        #d_all["metrics_std_avg"]["beatrate"] = 0
        d_all["metrics_max_std"]["beatrate"] = 0
        d_all["metrics_avg_std"]["beatrate"] = 0
        #d_all["metrics_std_std"]["beatrate"] = 0

    return d_all
