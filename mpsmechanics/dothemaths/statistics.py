"""

Åshild Telle / Simula Research Laboratory / 2019

"""

import numpy as np

from mpsmechanics.dothemaths.heartbeat import \
        calc_beat_maxima, calc_beat_intervals


def calc_for_each_key(init_data, fun, filter_map):
    """

    Performs a given operations for a dictionary with similar data
    sets.

    Args:
        init_data - dictionary with attributes as keys, T x R values,
            where T is an integer; R an integer or tuple of integers
        fun - function f : R -> S

    Returns:
        A numpy array with resulting values, of dimension T x R

    """

    d_values = {}

    for key in init_data.keys():
        d_values[key] = fun(init_data[key], filter_map[key])

    return d_values


def fun_mean(org_values, filter_x):
    """
    Finds mean over filtered values.
    """
    all_values = np.zeros(org_values.shape[0])

    for time_step in range(org_values.shape[0]):
        values = np.extract(filter_x[time_step], \
                            org_values[time_step])
        all_values[time_step] = np.mean(values)

    return all_values


def fun_std(org_values, filter_x):
    """
    Finds std over filtered values.
    """
    all_values = np.zeros(org_values.shape[0])

    for time_step in range(org_values.shape[0]):
        values = np.extract(filter_x[time_step],
                            org_values[time_step])
        all_values[time_step] = np.std(values)

    return all_values


def calc_xmotion_metrics(avg, std, intervals):
    metr = {}

    intervals_avg = [np.mean(avg[i1:i2]) \
            for (i1, i2) in intervals]
    intervals_std = [np.mean(std[i1:i2]) \
            for (i1, i2) in intervals]

    metr["metrics_max_avg"] = np.max(intervals_avg)
    metr["metrics_avg_avg"] = np.mean(intervals_avg)
    metr["metrics_max_std"] = np.max(intervals_std)
    metr["metrics_avg_std"] = np.mean(intervals_std)

    return metr


def chip_statistics(data):
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
    fun_folded = lambda x, _: np.linalg.norm(x, axis=-1)
    fun_max = lambda x, _: max(x)

    d_all_keys = ["all_values", "units", "filters", "range"]
    d_keys = data.keys()

    for (i, all_key) in enumerate(d_all_keys):
        d_all[all_key] = {}
        for d_key in d_keys:
            d_all[all_key][d_key] = data[d_key][i]

    quantities = d_all["all_values"]
    time_filter = d_all["filters"]
    d_all["folded"] = \
            calc_for_each_key(quantities, fun_folded, time_filter)

    d_all["over_time_avg"] = \
            calc_for_each_key(d_all["folded"], fun_mean, time_filter)
    d_all["over_time_std"] = \
            calc_for_each_key(d_all["folded"], fun_std, time_filter)

    # general variables
    max_diff_key = "displacement maximum difference"

    d_all["maxima"] = \
        calc_beat_maxima(d_all["over_time_avg"][max_diff_key])
    d_all["intervals"] = \
        calc_beat_intervals(d_all["over_time_avg"][max_diff_key])

    fun_meanmax = lambda x, _: np.mean([max(x[i1:i2]) \
            for (i1, i2) in d_all["intervals"]])
 
    # metrics
    if len(d_all["intervals"]) > 1:
        d_all["metrics_max_avg"] = \
                calc_for_each_key(d_all["over_time_avg"], \
                fun_max, time_filter)
        d_all["metrics_avg_avg"] = \
                calc_for_each_key(d_all["over_time_avg"], \
                fun_meanmax, time_filter)
        d_all["metrics_max_std"] = \
                calc_for_each_key(d_all["over_time_std"], \
                fun_max, time_filter)
        d_all["metrics_avg_std"] = \
                calc_for_each_key(d_all["over_time_std"], \
                fun_meanmax, time_filter)
    
        # special case for xmotion

        xmotion_metrics = calc_xmotion_metrics(d_all["over_time_avg"]["xmotion"],
                                               d_all["over_time_std"]["xmotion"],
                                               d_all["intervals"])

        for key in xmotion_metrics.keys():
            d_all[key]["xmotion"] = xmotion_metrics[key]

    else:
        for st_metric in ["metrics_max_avg", "metrics_avg_avg", \
                          "metrics_max_std", "metrics_avg_std"]:
            for k in quantities.keys():
                d_all[st_metric] = {}
                d_all[st_metric][k] = np.nan
    
    return d_all
