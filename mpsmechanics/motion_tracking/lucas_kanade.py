

from scipy.integrate import cumtrapz
import numpy as np
import dask.array as da

def lucas_kanade_np(im1, im2, win=2):
    assert im1.shape == im2.shape
    I_x = np.zeros(im1.shape)
    I_y = np.zeros(im1.shape)
    I_t = np.zeros(im1.shape)
    I_x[1:-1, 1:-1] = (im1[1:-1, 2:] - im1[1:-1, :-2]) / 2
    I_y[1:-1, 1:-1] = (im1[2:, 1:-1] - im1[:-2, 1:-1]) / 2
    I_t[1:-1, 1:-1] = im1[1:-1, 1:-1] - im2[1:-1, 1:-1]
    params = np.zeros(im1.shape + (5,))  # Ix2, Iy2, Ixy, Ixt, Iyt
    params[..., 0] = I_x * I_x  # I_x2
    params[..., 1] = I_y * I_y  # I_y2
    params[..., 2] = I_x * I_y  # I_xy
    params[..., 3] = I_x * I_t  # I_xt
    params[..., 4] = I_y * I_t  # I_yt
    del I_x, I_y, I_t
    cum_params = np.cumsum(np.cumsum(params, axis=0), axis=1)
    del params
    win_params = (
        cum_params[2 * win + 1 :, 2 * win + 1 :]
        - cum_params[2 * win + 1 :, : -1 - 2 * win]
        - cum_params[: -1 - 2 * win, 2 * win + 1 :]
        + cum_params[: -1 - 2 * win, : -1 - 2 * win]
    )
    del cum_params
    op_flow = np.zeros(im1.shape + (2,))
    det = win_params[..., 0] * win_params[..., 1] - win_params[..., 2] ** 2
    op_flow_x = np.where(
        det != 0,
        (
            win_params[..., 1] * win_params[..., 3]
            - win_params[..., 2] * win_params[..., 4]
        )
        / det,
        0,
    )
    op_flow_y = np.where(
        det != 0,
        (
            win_params[..., 0] * win_params[..., 4]
            - win_params[..., 2] * win_params[..., 3]
        )
        / det,
        0,
    )
    op_flow[win + 1 : -1 - win, win + 1 : -1 - win, 0] = op_flow_x[:-1, :-1]
    op_flow[win + 1 : -1 - win, win + 1 : -1 - win, 1] = op_flow_y[:-1, :-1]
    return op_flow


def lucas_kanade(mps_data, block_size=9):
    images = np.moveaxis(mps_data.frames, 2, 0)  # time to first dimension
    num_time_steps = images.shape[0]

    im1 = images[0]

    vel = np.zeros((num_time_steps - 1, im1.shape[0], im1.shape[1], 2))
    for i in range(1, num_time_steps):
        im1 = images[i-1]
        im2 = images[i]
        V = lucas_kanade_np(im1, im2, win=block_size)
        
        vel[i - 1, :, :, :] = V

    return vel


def calc_disp_lk(mps_data):

    vel_ = da.from_array(lucas_kanade(mps_data))
    
    # Subtract mean velocity
    vel = vel_ - np.mean(vel_, axis=0).compute()

    # Integrate velocity to get displacement
    disp = np.cumsum(vel, axis=0)

    print("Done lucas kanade")

    return disp

