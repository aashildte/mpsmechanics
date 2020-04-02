"""

Åshild Telle / Simula Research Laboratory / 2019

"""

import numpy as np


def calc_std_tf_filter(values_folded, tf_filter):
    """

    Calculate standard deviation over filtered values.

    Args:
        values_folded - T x X x Y numpy array, float values
        tf_filter - T x X x Y numpy array, True/False values

    Returns:
        numpy 1D array of length T, standard deviation over time

    """

    assert (
        values_folded.shape == tf_filter.shape
    ), "Error: Shape mismatch: {values_folded.shape}, {tf_filter.shape}"

    num_time_steps = len(values_folded)

    std = np.zeros(num_time_steps)

    for _t in range(num_time_steps):
        if np.any(values_folded[_t]):
            filtered_values = np.extract(
                tf_filter[_t], values_folded[_t]
            )
            std[_t] = np.std(filtered_values)
        else:
            std[_t] = 0

    return std


def calc_avg_tf_filter(values_folded, tf_filter):
    """

    Calculate average over filtered values.

    Args:
        values_folded - T x X x Y numpy array, float values
        tf_filter - T x X x Y numpy array, True/False values

    Returns:
        numpy 1D array of length T, average over time

    """

    assert (
        values_folded.shape == tf_filter.shape
    ), "Error: Shape mismatch: {values_folded.shape}, {tf_filter.shape}"

    num_time_steps = len(values_folded)

    avg = np.zeros(num_time_steps)

    for _t in range(num_time_steps):
        if np.any(values_folded[_t]):
            filtered_values = np.extract(
                tf_filter[_t], values_folded[_t]
            )
            avg[_t] = np.mean(filtered_values)
        else:
            avg[_t] = 0

    return avg


def filter_constrained(values, size):
    """

    Filter independent of time (same for all time steps); edges
    (as defined by distance of *size* from a pixel block with
    no displacement) not included.

    Args:
        values – numpy arrray
        size - distance for which values are set to False if
            close enough to a zero-valued coordinate

    Returns:
        True/False array depending on nonzero value or not; same shape

    """

    filter_org = np.any((values != 0), axis=(0, -1))

    filter_new = np.copy(filter_org)

    dim_x, dim_y = filter_org.shape

    for _x in range(dim_x):
        for _y in range(dim_y):
            if not filter_org[_x, _y]:
                xm2 = _x - size if _x > size else 0
                xp2 = _x + size if (_x + size) < dim_x else dim_x - 1

                ym2 = _y - size if _y > size else 0
                yp2 = _y + size if (_y + size) < dim_y else dim_y - 1
                filter_new[xm2 : xp2 + 1, ym2 : yp2 + 1] *= False

    return np.broadcast_to(filter_new, values.shape[:3])


def filter_time_dependent(values):
    """

    Filter dependent on time (different for all time steps).

    Args:
        values – numpy arrray

    Returns:
        True/False array depending on nonzero value or not;
            same shape as the original values array

    """
    return np.any((values != 0), axis=-1)


def filter_uniform(values):
    """

    Filter independent of time (same for all time steps); defined
    as if nonzero anywhere (for any time step, as taken to be the
    first dimension), then True everywhere.

    Args:
        values – numpy arrray

    Returns:
        True/False array depending on nonzero value or not;
            same shape as the original values array

    """

    return np.broadcast_to(
        np.any((values != 0), axis=(0, -1)), values.shape[:3]
    )
