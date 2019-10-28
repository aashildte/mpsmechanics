"""

Åshild Telle / Simula Research Laboratory / 2019

"""

import numpy as np
import mpsmechanics as mc


def test_calc_deformation_tensor():
    """

    Unit test for mpsmechanics/dothemaths/mechanical quantities
        -> calc_deformation_tensor

    """

    shape = (1, 4, 3, 2)
    data_org = np.zeros(shape)

    for x in range(shape[1]):
        for y in range(shape[2]):
            data_org[0, x, y, 0] = 0.2*x + 0.1*y
            data_org[0, x, y, 1] = 0.1*y

    data_exp = np.zeros(shape + (2,))
    tile = np.array(((1.2, 0.1), (0, 1.1)))

    for x in range(shape[1]):
        for y in range(shape[2]):
            data_exp[0, x, y] = tile

    assert np.allclose(data_exp, \
            mc.calc_deformation_tensor(data_org, 1))


def test_calc_gl_strain_tensor():
    """

    Unit test for mpsmechanics/dothemaths/mechanical quantities
        -> calc_gl_strain_tensor

    """
    shape = (1, 4, 3, 2)
    data_org = np.zeros(shape)

    for x in range(shape[1]):
        for y in range(shape[2]):
            data_org[0, x, y, 0] = 0.2*x + 0.1*y
            data_org[0, x, y, 1] = 0.1*y

    data_exp = np.zeros(shape + (2,))
    tile = np.array(((1.44, 0.55), (0.55, 1.21)))

    for x in range(shape[1]):
        for y in range(shape[2]):
            data_exp[0, x, y] = tile

    assert np.allclose(data_exp, \
            mc.calc_gl_strain_tensor(data_org, 1))


def test_calc_principal_strain():
    """

    Unit test for mpsmechanics/dothemaths/mechanical quantities
        -> calc_gl_strain_tensor

    """
    shape = (1, 4, 3, 2)
    data_org = np.zeros(shape)

    for x in range(shape[1]):
        for y in range(shape[2]):
            data_org[0, x, y, 0] = 0.2*x + 0.1*y
            data_org[0, x, y, 1] = 0.1*y

    data_exp = np.zeros(shape)

    strain_tensor = np.array(((1.44, 0.55), (0.55, 1.21)))
    [lambda1, _], [ev1, _] = np.linalg.eig(strain_tensor)

    for x in range(shape[1]):
        for y in range(shape[2]):
            data_exp[0, x, y] = lambda1*ev1

    assert np.allclose(data_exp, \
            mc.calc_principal_strain(data_org, 1))

if __name__ == "__main__":
    test_calc_deformation_tensor()
    test_calc_gl_strain_tensor()
    test_calc_principal_strain()
