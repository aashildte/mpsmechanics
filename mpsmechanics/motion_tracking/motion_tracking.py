#!/usr/bin/env python3
# c) 2001-2019 Simula Research Laboratory ALL RIGHTS RESERVED
#
# END-USER LICENSE AGREEMENT
# PLEASE READ THIS DOCUMENT CAREFULLY. By installing or using this
# software you agree with the terms and conditions of this license
# agreement. If you do not accept the terms of this license agreement
# you may not install or use this software.

# Permission to use, copy, modify and distribute any part of this
# software for non-profit educational and research purposes, without
# fee, and without a written agreement is hereby granted, provided
# that the above copyright notice, and this license agreement in its
# entirety appear in all copies. Those desiring to use this software
# for commercial purposes should contact Simula Research Laboratory AS:
# post@simula.no
#
# IN NO EVENT SHALL SIMULA RESEARCH LABORATORY BE LIABLE TO ANY PARTY
# FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES,
# INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE
# "MPS" EVEN IF SIMULA RESEARCH LABORATORY HAS BEEN ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE. THE SOFTWARE PROVIDED HEREIN IS
# ON AN "AS IS" BASIS, AND SIMULA RESEARCH LABORATORY HAS NO OBLIGATION
# TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
# SIMULA RESEARCH LABORATORY MAKES NO REPRESENTATIONS AND EXTENDS NO
# WARRANTIES OF ANY KIND, EITHER IMPLIED OR EXPRESSED, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY OR FITNESS
import time
import os
import logging
import hashlib
import concurrent.futures
from scipy import ndimage
import scipy.stats as st
import numpy as np

import mps

from mps import utils

# from mps import plotter

from ..utils.data_layer import save_dictionary, generate_filename
from ..utils.bf_mps import BFMPS


__author__ = "Henrik Finsberg (henriknf@simula.no), 2017--2019"
__maintainer__ = "Henrik Finsberg"
__email__ = "henriknf@simula.no"

"""

This module is based on the Matlab code in the script scripttotry1105_nd2.m
provided by Berenice

"""

logger = utils.get_logger(__name__)

try:
    from numba import jit
except ImportError:
    msg = (
        "numba not found - Numba is just to speed up the motion tracking algorithm\n"
        "To install numba use: pip install numba"
    )
    logger.warning(msg)

    # Create a dummy decorator
    class jit:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, f):
            return f


def block_matching_map(args):
    """
    
    Helper function for running block maching algorithm in paralell

    """
    vectors = block_matching(*args[:-1])
    filter_kernel_size = args[-1]

    if filter_kernel_size > 0:
        vectors[:, :, 0] = ndimage.median_filter(vectors[:, :, 0], filter_kernel_size)
        vectors[:, :, 1] = ndimage.median_filter(vectors[:, :, 1], filter_kernel_size)

    return vectors


@jit(nopython=True)
def block_matching(
        reference_image: np.ndarray,
        image: np.ndarray,
        block_size: int,
        max_block_movement: int,
):
    """
    Computes the displacements from `reference_image` to `image`
    using a block matching algorithm. Briefly, we subdivde the images
    into blocks of size `block_size x block_size` and compare the two images
    within a range of +/- max_block_movement for each block.

    Arguments
    ---------
    reference_image : np.ndarray
        The frame used as reference
    image : np.ndarray
        The frame that you want to compute displacement for relative to
        the referernce frame
    block_size : int
        Size of the blocks
    max_block_movement : int
        Maximum allowed movement of blocks when searching for best match.

    Note
    ----
    Make sure to have max_block_movement big enough. If this is too small
    then the results will be wrong. It is better to choose a too large value
    of this. However, choosing a too large value will mean that you need to
    compare more blocks which will increase the running time.
    """
    # Shape of the image that is returned
    y_size, x_size = image.shape
    shape = (y_size // block_size, x_size // block_size)
    vectors = np.zeros((shape[0], shape[1], 2))
    costs = np.ones((2 * max_block_movement + 1, 2 * max_block_movement + 1))

    # Need to copy images to float array
    # otherwise negative values will be converted to large 16-bit integers
    ref_block = np.zeros((block_size, block_size))  # Block for reference image
    block = np.zeros((block_size, block_size))  # Block for image

    # Loop over each block
    for y_block in range(shape[0]):
        for x_block in range(shape[1]):

            # Coordinates in the orignal imagee
            y_image = y_block * block_size
            x_image = x_block * block_size

            block[:] = image[
                y_image : y_image + block_size, x_image : x_image + block_size,
            ]

            # Check if box has values
            if np.max(block) > 0:

                # Loop over values around the block within the `max_block_movement` range
                for i, y_block_ref in enumerate(
                        range(-max_block_movement, max_block_movement + 1)
                ):
                    for j, x_block_ref in enumerate(
                            range(-max_block_movement, max_block_movement + 1,)
                    ):

                        y_image_ref = y_image + y_block_ref
                        x_image_ref = x_image + x_block_ref

                        # Just make sure that we are within the referece image
                        if (y_image_ref < 0
                                or y_image_ref + block_size > y_size
                                or x_image_ref < 0
                                or x_image_ref + block_size > x_size):
                            costs[i, j] = np.nan
                        else:
                            ref_block[:] = reference_image[
                                y_image_ref : y_image_ref + block_size,
                                x_image_ref : x_image_ref + block_size,
                            ]
                            # Could improve this cost function / template matching
                            costs[i, j] = np.sum(
                                np.abs(np.subtract(block, ref_block))
                            ) / (block_size ** 2)

                # Find minimum cost vector and store it
                dy, dx = np.where(costs == np.nanmin(costs))

                # If there are more then one minima then we select none
                if len(dy) > 1 or len(dx) > 1:
                    vectors[y_block, x_block, 0] = 0
                    vectors[y_block, x_block, 1] = 0
                else:
                    vectors[y_block, x_block, 0] = max_block_movement - dy[0]
                    vectors[y_block, x_block, 1] = max_block_movement - dx[0]
            else:
                # If no values in box set to no movement
                vectors[y_block, x_block, :] = 0

    return vectors


class MotionTracking(object):
    """
    Class for performing motion tracking on BrightField data, in order
    to obtain velo

    Arguments
    ---------
    mps_data : mps.MPS
        The data that you want to analyze
    block_size : float
        Sice of each block in pixels. Default value: 9.
    max_block_movement: float
        Maximum allowed movement of each block. Default value: 18.
    serial : bool
        Run motion tracking in serial. Default: False (i.e default is paralell).
        Useful for debugging
    filter_kernel_size: int
        Size of kernel for median filter applied to the motion vectors. Default: 8
    loglevel : int
        Level of how much that is printed (Default: 20 (INFO))
    """

    def __init__(self,
                 data: mps.MPS,
                 block_size: int = 9,
                 max_block_movement: int = 18,
                 serial: bool = False,
                 filter_kernel_size: int = 8,
                 loglevel: int = logging.INFO):

        global logger
        logger = utils.get_logger(__name__, loglevel)
        self.data = data
        self.block_size = block_size
        self.max_block_movement = max_block_movement
        self.serial = serial
        self.filter_kernel_size = filter_kernel_size

        logger.info(
            (
                "Initializing motion tracker with :\n"
                f"Block size: {self.block_size} pixels "
                f"Max movement: {self.max_block_movement} pixels "
            )
        )

        self.macro_shape = (
            data.frames.shape[0] // self.block_size,
            data.frames.shape[1] // self.block_size,
        )

        self.shape = (
            self.macro_shape[0] * self.block_size,
            self.macro_shape[1] * self.block_size,
        )

        from mps import __version__

        self._signature = hashlib.sha1(
            str(
                repr(data)
                + str(block_size)
                + str(max_block_movement)
                + str(__version__)
            ).encode("utf-8")
        ).hexdigest()

        self.displacement_vectors = self.get_displacements()


    def _get_displacements_iter(self):
        """
        Iterable to be passed in to the map fuction for displacements
        """

        reference = np.median(self.data.frames[: self.shape[0], : self.shape[1], :], 2,)

        def gen():
            for i in range(self.data.num_frames):
                yield (
                    reference,
                    self.data.frames[: self.shape[0], : self.shape[1], i],
                    self.block_size,
                    self.max_block_movement,
                    self.filter_kernel_size,
                )

        return gen()

    def get_displacements(self):

        logger.info("Get displacements")
        displacement_vectors = np.zeros(
            (self.macro_shape[0], self.macro_shape[1], 2, self.data.num_frames,)
        )

        iterable = self._get_displacements_iter()
        t_start = time.time()
        if self.serial:
            for i, vectors in enumerate(map(block_matching_map, iterable)):
                if i % 50 == 0:
                    logger.info(f"Processing frame {i}/{self.data.num_frames - 1}")
                displacement_vectors[:, :, :, i] = vectors
        else:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                for i, vectors in enumerate(executor.map(block_matching_map, iterable)):
                    if i % 50 == 0:
                        logger.info(f"Processing frame {i}/{self.data.num_frames - 1}")
                    displacement_vectors[:, :, :, i] = vectors

        t_end = time.time()
        logger.info(
            ("Done getting displacements " f" - Elapsed time = {t_end-t_start:.2f} seconds")
        )

        return displacement_vectors


def calc_chamber_angle(displacement_vectors: np.ndarray) -> float:
    """

    Estimating angle/how much the chamber is tilted using linear
    regression.

    Args:
        displacement_vectors : np array of dimension T x X x Y x 2

    Returns:
        estimated angle

    """

    assert len(displacement_vectors.shape) == 4 and \
            displacement_vectors.shape[-1] == 2, \
            "Error: Expected shape T x X x Y x 2"

    num_time_steps = displacement_vectors.shape[-1]

    x_values, y_values = [], []

    for _t in range(num_time_steps):
        xs_, ys_, _ = np.nonzero(displacement_vectors[:, :, :, _t])
        x_values += list(xs_)
        y_values += list(ys_)

    if not x_values:
        print("Warning: No nonzero values detected.")
        slope = 0
    else:
        slope = st.linregress(x_values, y_values)[0]

    return np.arctan(slope)


def track_motion(f_in, overwrite, overwrite_all, param_list, save_data=True):
    """

    Args:
        f_in - nd2 or zip file
        overwrite - recalculate values, or not
        overwrite_all - in this script, same as overwrite
        param_list - give parameters to motion tracking algorithm,
            predefined set of values
        save_data - boolean value: save as npy file when finished, or not

    Returns:
        dictionary with motion data and relevant information from
            mps file

    """

    filename = generate_filename(f_in, "track_motion", param_list, ".npy")

    if not (overwrite or overwrite_all) and os.path.isfile(filename):
        print("Previous data exist. Use flag --overwrite / -o to recalculate.")
        return np.load(filename, allow_pickle=True).item()

    np.seterr(invalid="ignore")

    mps_data = BFMPS(f_in)
    assert mps_data.num_frames != 1, "Error: Single frame used as input"

    if len(param_list) > 1:
        motion = MotionTracking(mps_data, **(param_list[0]))
    else:
        motion = MotionTracking(mps_data)

    # convert to T x X x Y x 2 - TODO maybe we can do this earlier actually
    disp_data = np.swapaxes(
        np.swapaxes(np.swapaxes(motion.displacement_vectors, 0, 1), 0, 2), 0, 3,
    )

    # save values

    d_all = {}
    d_all["displacement_vectors"] = disp_data
    d_all["angle"] = calc_chamber_angle(disp_data)
    d_all["block_size"] = motion.block_size
    d_all["info"] = mps_data.info

    print("Motion tracking done.")

    if save_data:
        save_dictionary(filename, d_all)

    return d_all
