"""

Åshild Telle / Simula Research Laboratory / 2019

"""

import numpy as np
from scipy.ndimage import gaussian_filter

from ..mechanical_analysis.filters import filter_constrained

def apply_filter(motion_data, type_filter, sigma):
    """

    Attempts to restore resolution which was lost due to downsampling
    to pixel (integer) values.

    Args:
        motion_data - T x X x Y x 2 numpy array
        type_filter - "gaussian" or "downsampling"
        sigma - argument to filter method, either
          * sigma in x and y direction for gaussian filter
          * number of pixel blocks to combine using downsampling
              approach

    Return:
        smoothered / downsampled data, similar shape as the original
            motion_data

    """

    assert type_filter in (
        "gaussian",
        "gaussian_mask",
        "downsampling",
    ), "Error_ Type filter not recognized."

    if type_filter == "gaussian":
        return gaussian_filter(motion_data, [0, sigma, sigma, 0])

    if type_filter == "gaussian_mask":
        mask = filter_constrained(motion_data, 5)[0]

        return gaussian_filter_with_mask(motion_data, sigma, mask)

    # else: downsamling
    sigma = int(sigma)
    t_dim, x_dim, y_dim, val_dim = motion_data.shape
    x_dim_d = x_dim // sigma
    y_dim_d = y_dim // sigma

    new_data = np.zeros((t_dim, x_dim_d, y_dim_d, val_dim))

    for _t in range(t_dim):
        for _x in range(x_dim_d):
            for _y in range(y_dim_d):
                for _d in range(val_dim):
                    avg = np.mean(
                        motion_data[
                            _t,
                            (sigma * _x) : (sigma * (_x + 1)),
                            (sigma * _y) : (sigma * (_y + 1)),
                            _d,
                        ]
                    )

                    new_data[_t, _x, _y, _d] = avg

    return new_data


def gaussian_filter_with_mask(
        motion_data: np.ndarray,
        sigma: float,
        mask: np.ndarray) -> np.ndarray:
    """

    Args:
        displacement/motion data – T x X x Y x 2 numpy array
        sigma - parameter for gaussian filter
        mask - which values to perform diffusion for;
            X x Y numpy array

    """

    assert len(motion_data.shape) == 4 and motion_data.shape[-1] == 2, \
        f"Error: Unexpected shape for motion data: {motion_data.shape};" + \
            " expected T x X x Y x 2"

    assert mask.shape == motion_data.shape[1:3], \
        f"Error: Unexpected shape for mask: {mask.shape};" + \
            f" expected {motion_data.shape[1:3]}"

    filtered_data = np.zeros_like(motion_data)

    for t in range(len(motion_data)):
        for i in range(2):
            data_loc = gaussian_filter(motion_data[t, :, :, i], sigma)
            data_loc[np.logical_not(mask)] = 0

            filtered_data[t, :, :, i] = \
                    np.where(mask, motion_data[t, :, :, i], data_loc)

    return filtered_data
