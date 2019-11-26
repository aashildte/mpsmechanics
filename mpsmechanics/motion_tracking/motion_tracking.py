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
__author__ = "Henrik Finsberg (henriknf@simula.no), 2017--2019"
__maintainer__ = "Henrik Finsberg"
__email__ = "henriknf@simula.no"
"""
This module is based on the Matlab code in the script scripttotry1105_nd2.m
provided by Berenice
"""
import time
import os
import logging
from pathlib import Path
from collections import namedtuple
import hashlib
import pickle
import itertools
import concurrent.futures
from scipy import ndimage
import scipy.stats as st
import numpy as np
from skimage.feature import match_template

import mps

from mps import utils

# from mps import plotter
from mps import analysis

from ..utils.iofuns.data_layer import save_dictionary, generate_filename

logger = utils.get_logger(__name__)
contraction_data_keys = [
    f"{key1}_{key2}"
    for key2 in ["max", "mean", "min", "std"]
    for key1 in ["x", "y", "amp", "angle"]
]
ContractionData = namedtuple("ContractionData", contraction_data_keys)


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


try:
    import skimage
    from skimage import feature, morphology, filters, exposure
    from skimage.restoration import denoise_wavelet, estimate_sigma
except ImportError:
    msg = (
        "skimage not found - motion tracking not possible\n"
        "Please install skimage in order to use the motion tracking algorithm"
        "\n    pip install scikit-image"
    )
    logger.warning(msg)


def block_matching_map(args):
    vectors = block_matching(*args[:-1])

    if args[-1] > 0:
        vectors[:, :, 0] = ndimage.median_filter(vectors[:, :, 0], args[-1])
        vectors[:, :, 1] = ndimage.median_filter(vectors[:, :, 1], args[-1])
    return vectors


def template_matching_map(args):
    vectors = template_matching(*args[:-1])

    if args[-1] > 0:
        vectors[:, :, 0] = ndimage.median_filter(vectors[:, :, 0], args[-1])
        vectors[:, :, 1] = ndimage.median_filter(vectors[:, :, 1], args[-1])
    return vectors


@jit(nopython=True)
def block_matching(reference_image, image, block_size, max_block_movement):
    """
    Computes the displacements from `reference_image` to `image`
    using a block matching algorthim. Briefly, we subdivde the images
    into blocks of size `block_size x block_size` and compare the two images
    within a range of +/- max_block_movement for each block.


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
                y_image : y_image + block_size, x_image : x_image + block_size
            ]

            # Check if box has values
            if np.max(block) > 0:

                # Loop over values around the block within the `max_block_movement` range
                for i, y_block_ref in enumerate(
                    range(-max_block_movement, max_block_movement + 1)
                ):
                    for j, x_block_ref in enumerate(
                        range(-max_block_movement, max_block_movement + 1)
                    ):

                        y_image_ref = y_image + y_block_ref
                        x_image_ref = x_image + x_block_ref

                        # Just make sure that we are within the referece image
                        if (
                            y_image_ref < 0
                            or y_image_ref + block_size > y_size
                            or x_image_ref < 0
                            or x_image_ref + block_size > x_size
                        ):
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


def template_matching(reference_image, image, block_size, max_block_movement):
    # Shape of the image that is returned
    y_size, x_size = image.shape
    shape = (y_size // block_size, x_size // block_size)
    vectors = np.zeros((shape[0], shape[1], 2))
    costs = np.ones((2 * max_block_movement + 1, 2 * max_block_movement + 1))

    # Need to copy images to float array
    # otherwise negative values will be converted to large 16-bit integers
    N = 2 * max_block_movement + block_size
    search_block = np.zeros((N, N))  # Block for reference image

    block = np.zeros((block_size, block_size))  # Block for image

    # Loop over each block
    for y_block in range(shape[0]):
        for x_block in range(shape[1]):

            # Coordinates in the orignal image
            y_image = y_block * block_size
            x_image = x_block * block_size

            block[:] = image[
                y_image : y_image + block_size, x_image : x_image + block_size
            ]

            if np.all(block == 0):
                # If no values in box set to no movement
                vectors[y_block, x_block, :] = 0
                continue

            y_min = max(0, y_image - max_block_movement)
            y_max = min(y_size, y_image + block_size + max_block_movement)
            x_min = max(0, x_image - max_block_movement)
            x_max = min(x_size, x_image + block_size + max_block_movement)

            search_block[:] = np.nan
            y_ref_start = y_min - y_image + max_block_movement
            y_ref_end = N - (y_image + block_size + max_block_movement - y_max)
            x_ref_start = x_min - x_image + max_block_movement
            x_ref_end = N - (x_image + block_size + max_block_movement - x_max)

            search_block[
                y_ref_start:y_ref_end, x_ref_start:x_ref_end
            ] = reference_image[y_min:y_max, x_min:x_max]

            result = match_template(search_block, block)
            dy, dx = np.where(np.abs(result - np.nanmax(result)) < 1e-5)
            # Select the case that has the smallest displacement
            idx = np.argmin(
                np.linalg.norm(np.abs(max_block_movement - np.array([dx, dy])), axis=0)
            )

            vectors[y_block, x_block, 0] = max_block_movement - dy[idx]
            vectors[y_block, x_block, 1] = max_block_movement - dx[idx]

    return vectors


def scale_to_macro_block(arr, block_size):

    logger.debug("Scale to macro blocks")
    shape = arr.shape
    macro_shape = (shape[0] // block_size, shape[1] // block_size)

    new_arr = np.zeros(shape, dtype=np.uint16)
    for k, l in itertools.product(np.arange(macro_shape[0]), np.arange(macro_shape[1])):
        b = np.uint16(
            arr[
                k * block_size : (k + 1) * block_size,
                l * block_size : (l + 1) * block_size,
            ]
        ).prod()
        new_arr[
            k * block_size : (k + 1) * block_size, l * block_size : (l + 1) * block_size
        ] = b
    return new_arr


def edge_detection(im):
    """
    Perform edge detection on image `im` and scale
    the edges according to macro blocks
    """

    if skimage.__version__ >= "0.15":
        msg = (
            "Edge detection requires another version of skimage (you have {})"
            'You need version < 0.15 - pip install "scikit-image<0.15"'
            ""
        ).format(skimage.__version__)
        logger.warning(msg)

    if isinstance(im, tuple):
        im = im[0]
    shape = im.shape
    imgI = np.zeros(shape, dtype=np.uint16)
    imgI[:] = im

    # Apply some filters to mimic the edge detection algorithm in Matlab
    logger.debug("Apply edge detector")
    img = utils.normalize_frames(imgI)
    gaussian = filters.gaussian(img)
    sobel = utils.normalize_frames(filters.sobel(gaussian))
    # Get edges
    local_edges = feature.canny(sobel)
    # Dilate edges
    logger.debug("Apply dilation")
    diamond = morphology.diamond(2)
    local_edges_diamond = morphology.dilation(local_edges, diamond)

    # Fill the holes
    logger.debug("Fill holes")
    seed = np.copy(local_edges_diamond)
    seed[1:-1, 1:-1] = 1.0
    filled = morphology.reconstruction(seed, local_edges_diamond, method="erosion")
    # Remove small objects that have fewer than 0.2 percent of the whole size
    min_size = int(img.shape[0] * img.shape[1] * 0.002)
    logger.debug(f"Remove objects smaller then {min_size} pixels")
    cleaned = morphology.remove_small_objects(filled.astype(bool), min_size=min_size)

    return cleaned


def mean_contraction(amplitude, vectors, um_per_pixel, factor=1.0):

    amplitude_map = np.sum(amplitude, 2)
    mask = amplitude_map < np.median(amplitude_map)

    results = {k: np.zeros(amplitude.shape[:2]) for k in contraction_data_keys}

    mapper = dict(
        amp=amplitude, x=vectors[:, :, 1, :], y=vectors[:, :, 0, :], angle=None
    )

    for key in results.keys():
        key1, key2 = key.split("_")
        data = mapper[key1]
        if data is None:
            continue

        data[mask] = 0
        value = getattr(np, key2)(data, 2)
        value *= um_per_pixel * factor
        results[key][:] = value

    return ContractionData(**results)


def contraction_data(amplitude, vectors, um_per_pixel, factor=1.0):

    amplitude_map = np.sum(amplitude, 2)
    mask = amplitude_map < np.median(amplitude_map)

    num_frames = amplitude.shape[-1]
    results = {k: np.zeros(num_frames) for k in contraction_data_keys}

    data = dict(
        x=np.zeros_like(vectors[:, :, 0, 0]),
        y=np.zeros_like(vectors[:, :, 1, 0]),
        amp=np.zeros_like(amplitude[:, :, 0]),
        angle=np.zeros_like(amplitude[:, :, 0]),
    )

    for t in range(num_frames):

        data["y"] = vectors[:, :, 0, t]
        # Add minus here for some reason
        data["x"] = -vectors[:, :, 1, t]
        data["amp"] = amplitude[:, :, t]

        data["y"][mask] = np.nan
        data["x"][mask] = np.nan
        data["amp"][mask] = np.nan
        data["angle"] = np.rad2deg(
            np.arctan(
                np.divide(
                    data["x"],
                    data["y"],
                    where=(data["y"] != 0),
                    out=np.zeros_like(data["x"]),
                )
            )
        )

        for key in results.keys():
            key1, key2 = key.split("_")
            v = data[key1]
            # Use the nan function from numpy
            value = getattr(np, "nan" + key2)(v)
            if key1 != "angle":
                value *= um_per_pixel * factor
            results[key][t] = value

    return ContractionData(**results)


def subtract_frame(img, frame="0"):
    """
    Remove frame from all other frames
    """

    f = np.zeros_like(img[0, 0, :])
    if frame.isnumeric():
        idx = int(frame)
        f = img[:, :, idx]

    return np.subtract(img, np.repeat(f, img.shape[-1]).reshape(img.shape))


class MotionTracking(object):
    """
    Class for performing motion tracking on BrightField data, in order
    to obtain velo

    Arguments
    ---------
    mps_data : mps.MPS
        The data that you want to analyze
    block_size : float
        Sice of each block in micrometers. Default: 3 micrometers
    max_block_movement: float
        Maximum allowed movement of each block in micrometers
        Default 3 micrometers.
    matching_method : str
        Method used for block matching. Choices "block_matching" (default) or
        "template_matching" 
    """

    _arrays = ["velocity_vectors", "displacement_vectors"]

    def __init__(
        self,
        data,
        block_size=3,
        max_block_movement=3,
        reference_frame="median",
        delay=None,
        outdir=None,
        serial=False,
        filter_kernel_size=8,
        matching_method="block_matching",
        loglevel=logging.INFO,
    ):

        global logger
        logger = utils.get_logger(__name__, loglevel)
        self.data = data
        self.block_size_microns = block_size
        self.block_size = int(block_size / data.info["um_per_pixel"])
        self.max_block_movement_microns = max_block_movement
        self.max_block_movement = int(max_block_movement / data.info["um_per_pixel"])
        self.serial = serial
        self.reference_frame = reference_frame
        self.filter_kernel_size = filter_kernel_size

        self.matching_map = (
            block_matching_map
            if matching_method == "block_matching"
            else template_matching_map
        )

        logger.info(
            (
                "Initializing motion tracker with :\n"
                f"Block size: {self.block_size} pixels "
                f"/ {self.block_size_microns} um \n"
                f"Max movement: {self.max_block_movement} pixels "
                f"/ {self.max_block_movement_microns} um \n"
            )
        )
        if delay is None:
            delay = int(np.ceil(0.1 * data.framerate))
        self.delay = delay

        self.macro_shape = (
            data.frames.shape[0] // self.block_size,
            data.frames.shape[1] // self.block_size,
        )
        self.N = data.num_frames - self.delay
        self.shape = (
            self.macro_shape[0] * self.block_size,
            self.macro_shape[1] * self.block_size,
        )
        self._init_arrays()

        if outdir is None:
            if hasattr(data, "_fname"):
                outdir = Path(os.path.splitext(data._fname)[0])
            elif hasattr(data, "name"):
                outdir = Path(os.path.splitext(data.name)[0])
            else:
                outdir = Path("mps_motion_tracking")
        else:
            outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        self.outdir = outdir

        from mps import __version__

        self._signature = hashlib.sha1(
            str(
                repr(data)
                + str(block_size)
                + str(max_block_movement)
                + str(delay)
                + str(reference_frame)
                + str(__version__)
            ).encode("utf-8")
        ).hexdigest()

    def _init_arrays(self):

        self.computed = dict(
            angle=False, edges=False, velocities=False, displacements=False
        )

    @property
    def _get_velocities_iter(self):
        """
        Iterable to be passed in to the map fuction for GetMotionBioFormat
        """

        def gen():
            for i in range(self.N):
                yield (
                    self.data.frames[: self.shape[0], : self.shape[1], i],
                    self.data.frames[: self.shape[0], : self.shape[1], i + self.delay],
                    self.block_size,
                    self.max_block_movement,
                    self.filter_kernel_size,
                )

        return gen()

    def _get_displacements_iter(self):
        """
        Iterable to be passed in to the map fuction for
        GetMotionBioFormatStrainInterval
        """

        def check_int(s):
            if s[0] in ("-", "+"):
                return s[1:].isdigit()
            return s.isdigit()

        if check_int(self.reference_frame):
            idx = int(self.reference_frame)
            reference = self.data.frames[: self.shape[0], : self.shape[1], idx]
        elif self.reference_frame == "mean":
            reference = np.mean(
                self.data.frames[: self.shape[0], : self.shape[1], :], 2
            )
        elif self.reference_frame == "median":
            reference = np.median(
                self.data.frames[: self.shape[0], : self.shape[1], :], 2
            )
        else:
            msg = (
                "Unknown reference_frame {self.reference_frame}. "
                'Expected an integer (as a string) of "mean" or "median"'
            )
            raise ValueError(msg)

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

    def _edge_detection(self):

        logger.info("Run edge detection")

        self._edges = np.zeros(
            (self.shape[0], self.shape[1], self.data.num_frames), dtype=np.uint16
        )

        iterable = (
            self.data.frames[: self.shape[0], : self.shape[1], i]
            for i in range(self.data.num_frames)
        )
        t0 = time.time()
        if self.serial:
            for i, e in enumerate(map(edge_detection, iterable)):
                if i % 50 == 0:
                    logger.info(f"Processing frame {i}/{self.data.num_frames}")
                self._edges[:, :, i] = scale_to_macro_block(e, self.block_size)

        else:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                for i, e in enumerate(executor.map(edge_detection, iterable)):
                    if i % 50 == 0:
                        logger.info(f"Processing frame {i}/{self.data.num_frames}")
                    self._edges[:, :, i] = scale_to_macro_block(e, self.block_size)

        t1 = time.time()
        logger.info(f"Done with edge detection - Elapsed time = {t1-t0:.2f} seconds")
        self.computed["edges"] = True

    def _get_velocities(self):

        logger.info("Get velocities")
        self._velocity_vectors = np.zeros(
            (self.macro_shape[0], self.macro_shape[1], 2, self.N)
        )

        iterable = self._get_velocities_iter
        t0 = time.time()
        if self.serial:
            for i, v in enumerate(map(self.matching_map, iterable)):
                if i % 50 == 0:
                    logger.info(f"Processing frame {i}/{self.N}")
                self._velocity_vectors[:, :, :, i] = v

        else:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                for i, v in enumerate(executor.map(self.matching_map, iterable)):

                    if i % 50 == 0:
                        logger.info(f"Processing frame {i}/{self.N}")
                    self._velocity_vectors[:, :, :, i] = v

        t1 = time.time()
        logger.info(f"Done getting velocities - Elapsed time = {t1-t0:.2f} seconds")
        self.computed["velocities"] = True

    def _get_displacements(self):

        logger.info("Get displacements")
        self._displacement_vectors = np.zeros(
            (self.macro_shape[0], self.macro_shape[1], 2, self.data.num_frames)
        )

        iterable = self._get_displacements_iter()
        t0 = time.time()
        if self.serial:
            for i, v in enumerate(map(self.matching_map, iterable)):
                if i % 50 == 0:
                    logger.info(f"Processing frame {i}/{self.data.num_frames - 1}")
                    self._displacement_vectors[:, :, :, i] = v
        else:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                for i, v in enumerate(executor.map(self.matching_map, iterable)):
                    if i % 50 == 0:
                        logger.info(f"Processing frame {i}/{self.data.num_frames - 1}")
                    self._displacement_vectors[:, :, :, i] = v

        t1 = time.time()
        logger.info(
            ("Done getting displacements " f" - Elapsed time = {t1-t0:.2f} seconds")
        )
        self.computed["displacements"] = True

    def _get_angle(self):
        """
        
        Estimating angle/how much the chamber is tilted using linear
        regression.
        
        """

        disp = self.displacement_vectors
        T = disp.shape[-1]

        xs, ys = [], []

        for t in range(T):
            xs_, ys_, _ = np.nonzero(disp[:, :, :, t])
            xs += list(xs_)
            ys += list(ys_)

        if not xs:
            print("Warning: No nonzero values detected.")
            slope = 0
        else:
            slope = st.linregress(xs, ys)[0]

        self._angle = np.arctan(slope)
        self.computed["angle"] = True

    @property
    def angle(self):
        if True or not self.has_results("angle"):
            self._get_angle()
        return np.copy(self._angle)

    @property
    def edges(self):
        if not self.has_results("edges"):
            self._edge_detection()
        return np.copy(self._edges)

    @property
    def displacement_vectors(self):
        if not self.has_results("displacements"):
            self._get_displacements()
        return np.copy(self._displacement_vectors)

    @property
    def displacement_amp(self):
        if not self.has_results("displacements"):
            self._get_displacements()
        return np.linalg.norm(self.displacement_vectors, axis=2)

    @property
    def velocity_vectors(self):
        if not self.has_results("velocities"):
            self._get_velocities()
        return np.copy(self._velocity_vectors)

    @property
    def velocity_amp(self):
        if not self.has_results("velocities"):
            self._get_velocities()
        return np.linalg.norm(self.velocity_vectors, axis=2)

    @property
    def mean_displacement(self):
        if not hasattr(self, "_mean_displacement"):
            self._mean_displacement = mean_contraction(
                amplitude=self.displacement_amp,
                vectors=self.displacement_vectors,
                um_per_pixel=self.data.info["um_per_pixel"],
                factor=1.0,
            )
        return self._mean_displacement

    @property
    def mean_velocity(self):
        if not hasattr(self, "_mean_velocity"):
            self._mean_velocity = mean_contraction(
                amplitude=self.velocity_amp,
                vectors=self.velocity_vectors,
                um_per_pixel=self.data.info["um_per_pixel"],
                factor=self.data.framerate / self.delay,
            )
        return self._mean_velocity

    @property
    def velocity_data(self):
        if not hasattr(self, "_velocity_data"):
            self._velocity_data = contraction_data(
                amplitude=self.velocity_amp,
                vectors=self.velocity_vectors,
                um_per_pixel=self.data.info["um_per_pixel"],
                factor=self.data.framerate / self.delay,
            )
        return self._velocity_data

    @property
    def displacement_data(self):
        if not hasattr(self, "_displacement_data"):
            self._displacement_data = contraction_data(
                amplitude=self.displacement_amp,
                vectors=self.displacement_vectors,
                um_per_pixel=self.data.info["um_per_pixel"],
                factor=1.0,
            )
        return self._displacement_data

    def plot_displacement_data(self, fname=None):
        if fname is None:
            fname = self.outdir.joinpath("displacement_data")
        logger.info("Plot displacement data")
        plotter.plot_contraction_data(
            contraction_data=self.displacement_data,
            time_stamps=self.data.time_stamps,
            time_unit=self.data.info["time_unit"],
            label="Micrometers",
            fname=fname,
        )

    def plot_velocity_data(self, fname=None):
        if fname is None:
            fname = self.outdir.joinpath("velocity_data")
        logger.info("Plot velocity data")
        plotter.plot_contraction_data(
            contraction_data=self.velocity_data,
            time_stamps=self.data.time_stamps,
            time_unit=self.data.info["time_unit"],
            label="Micrometers / millisecond",
            fname=fname,
        )

    def plot_mean_displacement(self, fname=None):
        if fname is None:
            fname = self.outdir.joinpath("mean_displacement")
        logger.info("Plot mean displacement")
        plotter.plot_mean_contraction(
            frames=self.data.frames,
            contraction_data=self.mean_displacement,
            label="Micrometers",
            fname=fname,
        )

    def plot_mean_velocity(self, fname=None):
        if fname is None:
            fname = self.outdir.joinpath("mean_velocity")
        logger.info("Plot mean velocity")
        plotter.plot_mean_contraction(
            frames=self.data.frames,
            contraction_data=self.mean_velocity,
            label="Micrometers / millisecond",
            fname=fname,
        )

    def plot_velocity_field(self, fname=None):
        if fname is None:
            fname = self.outdir.joinpath("velocity_field")
        logger.info("Plot velocity field")
        plotter.animate_vectorfield(
            vectors=self.velocity_vectors,
            images=self.data.frames,
            framerate=self.data.framerate,
            fname=fname,
        )

    def plot_displacement_field(self, fname=None):
        if fname is None:
            fname = self.outdir.joinpath("displacement_field")
        logger.info("Plot displacement field")
        plotter.animate_vectorfield(
            vectors=self.displacement_vectors,
            images=self.data.frames,
            framerate=self.data.framerate,
            fname=fname,
        )

    def has_results(self, key):
        """
        Check if key is in the results and that it is differnet from zero.
        We can you this to see if results loaded from the cache should be
        used or recomputed (assuming that results identical to zero is
        not possible).
        """
        return self.computed[key]

    def save_motion(self, fname=None):
        if fname is None:
            fname = self.outdir.joinpath("motion")
        utils.frames2mp4(self.data.frames.T, fname, self.data.framerate)

    def save_displacement(self, fname=None):

        logger.info("Save displacements to csv")
        header = ["t", "x", "y", "displacement_X", "displacement_Y"]

        if fname is None:
            fname = self.outdir.joinpath("displacement").as_posix()

        d = self.displacement_vectors
        N = d.shape[-1] * d.shape[0] * d.shape[1]
        res = np.zeros((N, 5))
        i = 0
        # This is really slow - we could vectorize it,
        # but do we really want to save this to csv?
        for t in range(d.shape[-1]):
            for y in range(d.shape[1]):
                for x in range(d.shape[0]):
                    res[i, :] = [t, x, y, d[x, y, 0, t], d[x, y, 1, t]]
                    i += 1

        utils.to_csv(res, fname, header)
        logger.info("Done saving displacements to csv")

    def plot_all(self):
        self.plot_displacement_data()
        self.plot_velocity_data()
        self.plot_mean_displacement()
        self.plot_mean_velocity()
        self.plot_velocity_field()
        self.plot_displacement_field()

        return True

    def run(self):
        """
        Run the full motion tracking algorithm.
        """
        if not self.has_results("velocities"):
            self._get_velocities()

        if not self.has_results("displacements"):
            self._get_displacements()

        return True

    def save_data(self, fname=None):
        if fname is None:
            fname = self.outdir.joinpath("motion_data.npy")
        np.save(
            fname,
            dict(
                displacement=utils.namedtuple2dict(self.displacement_data),
                velocity=utils.namedtuple2dict(self.velocity_data),
            ),
        )


def track_motion(f_in, overwrite, param_list, save_data=True):
    """

    Args:
        f_in - nd2 or zip file
        overwrite - recalculate values, or not
        param_list - give parameters to motion tracking algorithm, 
            predefined set of values
        save_data - boolean value: save as npy file when finished, or not

    Returns:
        dictionary with motion data and relevant information from
            mps file

    """

    filename = generate_filename(f_in, "track_motion", param_list) + ".npy"

    print("Parameters motion tracking:")
    for key in param_list[0].keys():
        print(" * {}: {}".format(key, param_list[0][key]))

    if not overwrite and os.path.isfile(filename):
        print("Previous data exist. Use flag --overwrite / -o to recalculate.")
        return np.load(filename, allow_pickle=True).item()

    np.seterr(invalid="ignore")
    mt_data = mps.MPS(f_in)

    assert mt_data.num_frames != 1, "Error: Single frame used as input"

    motion = MotionTracking(mt_data, **(param_list[0]))

    # convert to T x X x Y x 2 - TODO maybe we can do this earlier actually

    disp_data = np.swapaxes(np.swapaxes(np.swapaxes(motion.displacement_vectors, 0, 1), 0, 2), 0, 3)

    # save values

    d_all = {}
    d_all["displacement_vectors"] = disp_data
    d_all["angle"] = motion.angle
    d_all["block_size"] = int(param_list[0]["block_size"] / mt_data.info["um_per_pixel"])

    print("Motion tracking done.")

    if save_data:
        save_dictionary(filename, d_all)

    return d_all
