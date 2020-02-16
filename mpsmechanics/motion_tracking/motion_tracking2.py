
import numpy as np


def block_matching(
    reference_image: np.ndarray,
    image: np.ndarray,
    block_size: int,
    max_block_movement: int,
    coordinates: np.ndarray
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
    coordinates : np.ndarray
        coordinates (midpoints) of blocks to track, dimension N x 2 where
        N is the number of points, each of them with an x and y coordinate

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
    vectors = np.zeros_like(coordinates)
    costs = np.ones((2 * max_block_movement + 1, 2 * max_block_movement + 1))

    # Need to copy images to float array
    # otherwise negative values will be converted to large 16-bit integers
    ref_block = np.zeros((block_size, block_size))  # Block for reference image
    block = np.zeros((block_size, block_size))  # Block for image

    # Loop over each coordinate
    for h in range(len(coordinates)):
        y_coord, x_coord = coordinates[h]
        
        # Coordinates in the orignal image
        y_image = y_coord - (block_size // 2)
        x_image = x_coord - (block_size // 2)

        if y_image < 0 or x_image < 0 or \
                y_image + block_size > y_size or \
                x_image + block_size > x_size:
            vectors[h, 0] = 0
            vectors[h, 1] = 0
            continue

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
                vectors[h, 0] = 0
                vectors[h, 1] = 0
            else:
                vectors[h, 0] = max_block_movement - dy[0]
                vectors[h, 1] = max_block_movement - dx[0]
        else:
            # If no values in box set to no movement
            vectors[h,:] = 0

    return vectors


def _generate_init_coords(mps_data, block_size):
    edge = block_size // 2

    x_max = mps_data.size_x - edge
    y_max = mps_data.size_y - edge

    x_coords = np.arange(edge, x_max, block_size, dtype=int)
    y_coords = np.arange(edge, y_max, block_size, dtype=int)
    num_x_coords = len(x_coords)
    num_y_coords = len(y_coords)

    coords_shape = (num_x_coords*num_y_coords, 2)
    coords = np.zeros(coords_shape, dtype=int)

    cnt = 0
    for _x in x_coords:            # I'm sure there's a more efficient way to do this TODO
        for _y in y_coords:
            coords[cnt][0] = _x
            coords[cnt][1] = _y
            cnt += 1

    return coords


def calc_disp_vectors_interpolate(
        mps_data: np.ndarray,
        block_size: int,
        max_block_movement: int
        ):
    
    images = np.moveaxis(mps_data.frames, 2, 0)      # time to first dimension
    num_time_steps = images.shape[0]

    coords = _generate_init_coords(mps_data, block_size)

    init_coords = np.copy(coords)
    disp_vectors = np.zeros((num_time_steps, coords.shape[0], 2))

    # follow some ponints wherever; go backwards as we want to end up in first frame

    for _t in range(1, num_time_steps):
        prev_frame = images[num_time_steps - _t - 1]
        curr_frame = images[num_time_steps - _t]

        diff = block_matching(prev_frame, curr_frame, block_size, max_block_movement, coords)
        coords -= diff

    init_coords = np.copy(coords)
    
    # then use these to find displacement

    for _t in range(1, num_time_steps):
        prev_frame = images[_t - 1]
        curr_frame = images[_t]

        diff = block_matching(prev_frame, curr_frame, block_size, max_block_movement, coords)
        coords += diff
        disp_vectors[_t] = coords - init_coords

    return disp_vectors


def calc_disp_vectors_meshpoints(
        mps_data: np.ndarray,
        block_size: int,
        max_block_movement: int
        ):

    images = np.moveaxis(mps_data.frames, 2, 0)      # time to first dimension
    num_time_steps = images.shape[0]

    coords = _generate_init_coords(mps_data, block_size)

    init_coords = np.copy(coords)
    disp_vectors = np.zeros((num_time_steps, coords.shape[0], 2))

    for _t in range(1, num_time_steps):
        prev_frame = images[_t - 1]
        curr_frame = images[_t]

        diff = block_matching(prev_frame, curr_frame, block_size, max_block_movement, coords)
        coords -= diff
        disp_vectors[_t] = coords - init_coords

    return disp_vectors
