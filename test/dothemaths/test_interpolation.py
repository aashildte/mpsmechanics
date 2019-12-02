"""

Åshild Telle / Simula Research Laboratory / 2019

"""

import numpy as np
import mpsmechanics as mc



def test_interpolate_values_xy():
    """

    Unit test for mpsmechanics/dothemaths/interpolation
        -> interpolate_values_xy

    """

    shape = (6, 5, 2)
    x_coords = np.linspace(0, shape[0], shape[0])
    y_coords = np.linspace(0, shape[1], shape[1])
    org_data = np.zeros(shape)

    for _x in range(shape[0]):
        for _y in range(shape[1]):
            org_data[_x, _y, 0] = _x
            org_data[_x, _y, 1] = _y

    ip_fun = mc.interpolate_values_xy(x_coords, y_coords, org_data)

    assert ip_fun is not None


if __name__ == "__main__":
    test_interpolate_values_xy()
