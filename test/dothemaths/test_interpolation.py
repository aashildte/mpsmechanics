"""

Åshild Telle / Simula Research Laboratory / 2019

"""

import numpy as np
import mpsmechanics as mc



def test_interpolate_values_xy():
    """

    Unit (functionality) test for mpsmechanics/dothemaths/interpolation
        -> interpolate_values_xy

    """

    shape = (6, 5, 2)
    x_coords = np.linspace(0, shape[0], shape[0])
    y_coords = np.linspace(0, shape[1], shape[1])
    org_data = np.zeros(shape)
    
    for x in range(shape[0]):
        for y in range(shape[1]):
            org_data[x, y, 0] = x
            org_data[x, y, 1] = y

    fn = mc.interpolate_values_xy(x_coords, y_coords, org_data)

    assert fn is not None


if __name__ == "__main__": 
    test_interpolate_values_xy()
