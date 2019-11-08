"""

Åshild Telle / Simula Research Laboratory / 2019

"""

import numpy as np

import mpsmechanics as mc

def test_calc_projection():
    """

    Unit test for mpsmechanics/dothemaths/angular
        -> calc_projection_vectors

    """

    shape = (5, 4, 3, 1)
    alpha = np.pi/2
    data_org = np.tile(np.array((1, 1)), shape)
    data_exp = np.ones(shape[:3])

    assert np.allclose(data_exp,
                       mc.calc_projection(data_org, alpha))


def test_calc_projection_fraction():
    """

    Unit test for mpsmechanics/dothemaths/angular
        -> calc_projection_fractions

    """
 
    shape = (5, 4, 3, 1)
    alpha = np.pi/2
    data_org = np.tile(np.array((1, 1)), shape)
    data_exp = np.sqrt(2)/2*np.ones(shape[:3])

    assert np.allclose(data_exp,
                       mc.calc_projection_fraction(data_org, alpha))

"""
def test_calc_angle_diff():
    

    Unit test for mpsmechanics/dothemaths/angular
        -> calc_angle_diff

    

    shape = (5, 4, 3, 1)
    alpha = 0
    data_org = np.tile(np.array((1, 1)), shape)
    data_exp = (np.pi/4)*np.ones(shape)
 
    assert np.allclose(data_exp,
                       mc.calc_angle_diff(data_org, alpha))
"""

def test_flip_values():
    """

    Unit test for mpsmechanics/dothemaths/angular
        -> flip_values

    """

    shape = (5, 4, 3, 1)
    data_org = np.tile(np.array((1, -1)), shape)
    data_exp = np.tile(np.array((-1, 1)), shape)

    assert np.allclose(data_exp,
                       mc.flip_values(data_org))


if __name__ == "__main__":

    test_calc_projection()
    test_calc_projection_fraction()
    #test_calc_angle_diff()
    test_flip_values()
