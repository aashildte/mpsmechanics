
import numpy as np

import mpsmechanics as mc

def test_gaussian_filter_with_mask():
    """

    Unit test for mpsmechanics/motion_tracking/restore_resolution
        -> gaussian_filter_with_mask

    """

    shape = (1, 5, 5, 2)

    data_xy = np.array([[0, 1, 1, 0, 0],
                        [0, 1, 1, 1, 1],
                        [1, 1, 0, 1, 1],
                        [1, 1, 1, 1, 0],
                        [0, 0, 1, 1, 0]])

    data_org = np.zeros(shape)
    data_org[:, :, :, 0] = data_xy
    data_org[:, :, :, 1] = data_xy
    sigma = 3

    mask = np.asarray(data_xy.copy(), dtype=np.bool)

    # check if they get the same nonzeros

    data_exp = data_org != 0
    data_calc = mc.gaussian_filter_with_mask(data_org, sigma, mask) != 0

    assert np.all(np.equal(data_exp, data_calc))

    # check if they get the same zeros

    data_exp = data_org == 0
    data_calc = mc.gaussian_filter_with_mask(data_org, sigma, mask) == 0

    assert np.all(np.equal(data_exp, data_calc))

if __name__ == "__main__":

    test_gaussian_filter_with_mask()
