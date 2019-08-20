"""

Åshild Telle / Simula Research Laboratory / 2019

"""

import numpy as np

import mpsmechanics as mc

def test_calc_projection_vectors():
    """

    Unit test for mpsmechanics/dothemaths/angular
        -> calc_projection_vectors

    """

    alpha = np.pi/2
    shape = (2, 2, 2, 1)
    data_org = np.tile(np.array((1, 1)), shape)
    data_exp = np.tile(np.array((0, 1)), shape)

    assert np.allclose(data_exp,
                       mc.calc_projection_vectors(data_org, alpha))


def test_calc_projection_fraction():
    """

    Unit test for mpsmechanics/dothemaths/angular
        -> calc_projection_fractions

    """

    alpha = np.pi/2
    shape = (2, 2, 2, 1)
    data_org = np.tile(np.array((1, 1)), shape)
    data_exp = np.sqrt(2)/2*np.ones(shape)

    assert np.allclose(data_exp,
                       mc.calc_projection_fraction(data_org, alpha))


def test_flip_values():
    """

    Unit test for mpsmechanics/dothemaths/angular
        -> flip_values

    """

    shape = (2, 2, 2, 1)
    data_org = np.tile(np.array((1, -1)), shape)
    data_exp = np.tile(np.array((-1, 1)), shape)

    assert np.allclose(data_exp,
                       mc.flip_values(data_org))


def test_angular():
    """

    All unit tests for mpsmechanics/dothemaths/angular

    """

    test_calc_projection_vectors()
    test_calc_projection_fraction()
    test_flip_values()


if __name__ == "__main__":
    test_angular()
