"""

Åshild Telle / Simula Research Laboratory / 2019

"""

import numpy as np
import mpsmechanics as mc


def test_calc_gradients_stretch():
    """

    Unit test for mpsmechanics/dothemaths/mechanical quantities
        -> calc_deformation_tensor

    """
    shape = (1, 4, 3, 2)
    data_org = np.zeros(shape)

    for _x in range(shape[1]):
        for _y in range(shape[2]):
            data_org[0, _x, _y, 0] = 0.1*_x

    data_exp = np.zeros(shape + (2,))
    tile_exp = np.array(((0.1, 0), (0, 0)))

    for _x in range(shape[1]):
        for _y in range(shape[2]):
            data_exp[0, _x, _y] = tile_exp

    assert np.allclose(data_exp, \
                       mc.calc_gradients(data_org, 1))


def test_calc_gradients_shear():
    """

    Unit test for mpsmechanics/dothemaths/mechanical quantities
        -> calc_deformation_tensor

    """

    shape = (1, 4, 3, 2)
    data_org = np.zeros(shape)

    for _x in range(shape[1]):
        for _y in range(shape[2]):
            data_org[0, _x, _y, 1] = 0.1*_x

    data_exp = np.zeros(shape + (2,))
    tile_exp = np.array(((0, 0), (0.1, 0)))

    for _x in range(shape[1]):
        for _y in range(shape[2]):
            data_exp[0, _x, _y] = tile_exp

    assert np.allclose(data_exp, \
                        mc.calc_gradients(data_org, 1))


def test_calc_deformation_tensor_stretch():
    """

    Unit test for mpsmechanics/dothemaths/mechanical quantities
        -> calc_deformation_tensor

    """
    shape = (1, 4, 3, 2)
    data_org = np.zeros(shape)

    for _x in range(shape[1]):
        for _y in range(shape[2]):
            data_org[0, _x, _y, 0] = 0.1*_x

    data_exp = np.zeros(shape + (2,))
    tile_exp = np.array(((1.1, 0), (0, 1)))

    for _x in range(shape[1]):
        for _y in range(shape[2]):
            data_exp[0, _x, _y] = tile_exp

    assert np.allclose(data_exp, \
                       mc.calc_deformation_tensor(data_org, 1))


def test_calc_deformation_tensor_shear():
    """

    Unit test for mpsmechanics/dothemaths/mechanical quantities
        -> calc_deformation_tensor

    """

    shape = (1, 4, 3, 2)
    data_org = np.zeros(shape)

    for _x in range(shape[1]):
        for _y in range(shape[2]):
            data_org[0, _x, _y, 1] = 0.1*_x

    data_exp = np.zeros(shape + (2,))
    tile_exp = np.array(((1, 0), (0.1, 1)))

    for _x in range(shape[1]):
        for _y in range(shape[2]):
            data_exp[0, _x, _y] = tile_exp

    assert np.allclose(data_exp, \
                       mc.calc_deformation_tensor(data_org, 1))


def test_calc_gl_strain_tensor_stretch():
    """

    Unit test for mpsmechanics/dothemaths/mechanical quantities
        -> calc_gl_strain_tensor

    """
    shape = (1, 4, 3, 2)
    data_org = np.zeros(shape)

    for _x in range(shape[1]):
        for _y in range(shape[2]):
            data_org[0, _x, _y, 0] = 0.1*_x

    data_exp = np.zeros(shape + (2,))
    tile_def_tensor = np.array(((1.1, 0), (0, 1)))
    tile_exp = 0.5*(np.matmul(tile_def_tensor, \
                              tile_def_tensor.T) - np.eye(2))

    for _x in range(shape[1]):
        for _y in range(shape[2]):
            data_exp[0, _x, _y] = tile_exp

    def_tensor = mc.calc_deformation_tensor(data_org, 1)

    assert np.allclose(data_exp, \
                       mc.calc_gl_strain_tensor(def_tensor))


def test_calc_gl_strain_tensor_shear():
    """

    Unit test for mpsmechanics/dothemaths/mechanical quantities
        -> calc_gl_strain_tensor

    """
    shape = (1, 4, 3, 2)
    data_org = np.zeros(shape)

    for _x in range(shape[1]):
        for _y in range(shape[2]):
            data_org[0, _x, _y, 1] = 0.1*_x

    data_exp = np.zeros(shape + (2,))
    tile_def_tensor = np.array(((1, 0), (0.1, 1)))
    tile_exp = 0.5*(np.matmul(tile_def_tensor, \
                              tile_def_tensor.T) - np.eye(2))

    for _x in range(shape[1]):
        for _y in range(shape[2]):
            data_exp[0, _x, _y] = tile_exp

    def_tensor = mc.calc_deformation_tensor(data_org, 1)

    assert np.allclose(data_exp, \
                       mc.calc_gl_strain_tensor(def_tensor))


def test_calc_principal_strain_stretch():
    """

    Unit test for mpsmechanics/dothemaths/mechanical quantities
        -> calc_gl_strain_tensor

    """
    shape = (1, 4, 3, 2)
    data_org = np.zeros(shape)

    for _x in range(shape[1]):
        for _y in range(shape[2]):
            data_org[0, _x, _y, 0] = 0.1*_x

    data_exp = np.zeros(shape)

    tile_def_tensor = np.array(((1.1, 0), (0, 1)))
    tile_gls_tensor = 0.5*(np.matmul(tile_def_tensor, \
                                     tile_def_tensor.T) - np.eye(2))
    [lambda1, _], [ev1, _] = np.linalg.eig(tile_gls_tensor)

    for _x in range(shape[1]):
        for _y in range(shape[2]):
            data_exp[0, _x, _y] = lambda1*ev1
    
    def_tensor = mc.calc_deformation_tensor(data_org, 1)
    gl_strain_tensor = mc.calc_gl_strain_tensor(def_tensor)

    assert np.allclose(data_exp, \
                       mc.calc_principal_strain(gl_strain_tensor, 1))


def test_calc_principal_strain_shear():
    """

    Unit test for mpsmechanics/dothemaths/mechanical quantities
        -> calc_gl_strain_tensor

    """
    shape = (1, 4, 3, 2)
    data_org = np.zeros(shape)

    for _x in range(shape[1]):
        for _y in range(shape[2]):
            data_org[0, _x, _y, 1] = 0.1*_x

    data_exp = np.zeros(shape)

    tile_def_tensor = np.array(((1, 0), (0.1, 1)))
    tile_gls_tensor = 0.5*(np.matmul(tile_def_tensor, \
                                     tile_def_tensor.T) - np.eye(2))
    lambdas, evs = np.linalg.eig(tile_gls_tensor)

    if lambdas[0] > lambdas[1]:
        pr_str = lambdas[0]*evs[0]
    else:
        pr_str = lambdas[1]*evs[1]

    for _x in range(shape[1]):
        for _y in range(shape[2]):
            data_exp[0, _x, _y] = pr_str

    def_tensor = mc.calc_deformation_tensor(data_org, 1)
    gl_strain_tensor = mc.calc_gl_strain_tensor(def_tensor)

    assert np.allclose(data_exp, \
                       mc.calc_principal_strain(gl_strain_tensor, 1))


if __name__ == "__main__":
    test_calc_gradients_stretch()
    test_calc_deformation_tensor_stretch()
    test_calc_gl_strain_tensor_stretch()
    test_calc_principal_strain_stretch()

    test_calc_gradients_shear()
    test_calc_deformation_tensor_shear()
    test_calc_gl_strain_tensor_shear()
    test_calc_principal_strain_shear()
