# -*- coding: utf-8 -*-
"""

Computes mechanical quantities / metrics over space and time.

Åshild Telle / Simula Research Labratory / 2019

"""

import numpy as np


from ..dothemaths.mechanical_quantities import \
        calc_principal_strain, calc_gl_strain_tensor, \
        calc_deformation_tensor
from ..dothemaths.angular import calc_projection_fraction
from ..dothemaths.operations import calc_magnitude

from ..motion_tracking.ref_frame import convert_disp_data, \
        calculate_minmax

from .filters import calc_avg_tf_filter, calc_std_tf_filter, \
        filter_time_dependent, filter_constrained, filter_uniform

def _calc_relevant_stats(values, intervals, tf_filter):

    folded = calc_magnitude(values)

    over_time_avg = calc_avg_tf_filter(folded, tf_filter)
    over_time_std = calc_std_tf_filter(folded, tf_filter)

    if len(intervals) > 1:
        intervals_avg = [np.max(over_time_avg[i1:i2]) \
                for (i1, i2) in intervals]
        intervals_std = [np.max(over_time_std[i1:i2]) \
                for (i1, i2) in intervals]

        metrics_max_avg = np.max(intervals_avg)
        metrics_avg_avg = np.mean(intervals_avg)
        metrics_max_std = np.max(intervals_std)
        metrics_avg_std = np.mean(intervals_std)
    else:
        metrics_max_avg = np.nan
        metrics_avg_avg = np.nan
        metrics_max_std = np.nan
        metrics_avg_std = np.nan


    return {"all_values" : values,
            "folded" : folded,
            "over_time_avg" : over_time_avg,
            "over_time_std" : over_time_std,
            "metrics_max_avg" : metrics_max_avg,
            "metrics_avg_avg" : metrics_avg_avg,
            "metrics_max_std" : metrics_max_std,
            "metrics_avg_std" : metrics_avg_std}


def _calc_displacement(displacement, intervals, tf_filter):

    statistical_qts = _calc_relevant_stats(displacement, \
                                           intervals, \
                                           tf_filter)
    metadata = {"unit" : r"$\mu m$",
                "range" : (-np.nan, np.nan),
                "range_folded" : (0, np.nan)}

    return {**statistical_qts, **metadata}


def _calc_displacement_minmax(displacement, intervals, tf_filter):

    displacement_minmax = convert_disp_data(
        displacement, calculate_minmax(displacement)
    )

    statistical_qts = _calc_relevant_stats(displacement_minmax, \
                                           intervals, \
                                           tf_filter)

    metadata = {"unit" : r"$\mu m$",
                "range" : (-np.nan, np.nan),
                "range_folded" : (0, np.nan)}

    return {**statistical_qts, **metadata}



def _calc_xmotion(displacement, angle, intervals, tf_filter):

    xmotion = calc_projection_fraction(displacement, angle)
    statistical_qts = _calc_relevant_stats(xmotion, \
                                           intervals, \
                                           tf_filter)

    metadata = {"unit" : "-",
                "range" : (0, 1),
                "range_folded" : (0, 1)}

    return {**statistical_qts, **metadata}


def _calc_ymotion(displacement, angle, intervals, tf_filter):

    ymotion = calc_projection_fraction(displacement, np.pi/2 + angle)
    statistical_qts = _calc_relevant_stats(ymotion, \
                                           intervals, \
                                           tf_filter)

    metadata = {"unit" : "-",
                "range" : (0, 1),
                "range_folded" : (0, 1)}

    return {**statistical_qts, **metadata}


def _calc_velocity(displacement, time, intervals, tf_filter):

    ms_to_s = 1e3
    velocity = ms_to_s * np.divide(
        np.gradient(displacement, axis=0), \
        np.gradient(time)[:, None, None, None]
    )

    statistical_qts = _calc_relevant_stats(velocity, \
                                           intervals, \
                                           tf_filter)

    metadata = {"unit" : r"$\mu m / s$",
                "range" : (-np.nan, np.nan),
                "range_folded" : (0, np.nan)}

    return {**statistical_qts, **metadata}


def _calc_prevalence(velocity_norm, intervals, tf_filter):

    threshold = 2      # um/s
    prevalence = np.where(velocity_norm > threshold*np.ones(velocity_norm.shape),
                          np.ones(velocity_norm.shape), np.zeros(velocity_norm.shape))

    statistical_qts = _calc_relevant_stats(prevalence, \
                                           intervals, \
                                           tf_filter)
    metadata = {"unit" : "-",
                "range" : (0, 1),
                "range_folded" : (0, 1)}

    return {**statistical_qts, **metadata}


def _calc_deformation_tensor(displacement, dx, intervals, tf_filter):

    deformation_tensor = calc_deformation_tensor(displacement, dx)

    statistical_qts = _calc_relevant_stats(deformation_tensor, \
                                           intervals, \
                                           tf_filter)
    metadata = {"unit" : r"-",
                "range" : (-np.nan, np.nan),
                "range_folded" : (-np.nan, np.nan)}

    return {**statistical_qts, **metadata}


def _calc_gl_strain_tensor(deformation_tensor, intervals, tf_filter):

    gl_strain_tensor = calc_gl_strain_tensor(deformation_tensor)

    statistical_qts = _calc_relevant_stats(gl_strain_tensor, \
                                           intervals, \
                                           tf_filter)
    metadata = {"unit" : r"-",
                "range" : (-np.nan, np.nan),
                "range_folded" : (0, np.nan)}

    return {**statistical_qts, **metadata}


def _calc_principal_strain(gl_strain_tensor, intervals, tf_filter):

    principal_strain = calc_principal_strain(gl_strain_tensor)

    statistical_qts = _calc_relevant_stats(principal_strain, \
                                           intervals, \
                                           tf_filter)
    metadata = {"unit" : r"-",
                "range" : (-np.nan, np.nan),
                "range_folded" : (0, np.nan)}

    return {**statistical_qts, **metadata}



def calc_spatial_metrics(displacement, time, dx, angle, intervals):
    """

    Derived quantities - reshape to match expected data structure
    for derived layers

    Args:
        displacement - displacement data, T x X x Y x 2 numpy array
        scale - scaling factor (pixels to um); dx
        angle - angle chamber is tilted with
        time - all time steps

    Returns:
        dictionary with relevant spatial quantities; each key gives
           a description, and each value is again a dictionary with
           a consistent set of representations of the given quantity

    """
    strain_filter_size = 4

    tf_filter_uniform = filter_uniform(displacement)
    tf_filter_timedep = filter_time_dependent(displacement)
    tf_filter_constrd = filter_constrained(displacement, \
            strain_filter_size)

    displacement_d = _calc_displacement(displacement, intervals, \
                                        tf_filter_uniform)

    disp_max_diff_d = _calc_displacement_minmax(displacement, intervals, \
                                                tf_filter_uniform)

    velocity_d = _calc_velocity(displacement, time, intervals, \
                                tf_filter_uniform)

    xmotion_d = _calc_xmotion(displacement, angle, intervals, \
                              tf_filter_timedep)
    
    ymotion_d = _calc_ymotion(displacement, angle, intervals, \
                              tf_filter_timedep)

    prevalence_d = _calc_prevalence(velocity_d["folded"], intervals, \
                                    tf_filter_uniform)

    def_tensor_d = _calc_deformation_tensor(displacement, dx, \
                                            intervals, tf_filter_constrd)

    gl_strain_tensor_d = _calc_gl_strain_tensor(def_tensor_d["all_values"], \
                                          intervals, tf_filter_constrd)

    pr_strain_d = _calc_principal_strain(gl_strain_tensor_d["all_values"], intervals, \
                                         tf_filter_constrd)

    metr = {"displacement" : displacement_d,
            "displacement_max_difference" : disp_max_diff_d,
            "velocity" : velocity_d,
            "xmotion" : xmotion_d,
            "ymotion" : ymotion_d,
            "prevalence" : prevalence_d,
            "deformation_tensor" : def_tensor_d,
            "Green-Lagrange_strain_tensor" : gl_strain_tensor_d,
            "principal_strain" : pr_strain_d}

    return metr
