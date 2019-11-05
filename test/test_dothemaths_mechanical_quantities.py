"""

Åshild Telle / Simula Research Laboratory / 2019

"""

import numpy as np
import mpsmechanics as mc


def test_calc_deformation_tensor_stretch():
    """

    Unit test for mpsmechanics/dothemaths/mechanical quantities
        -> calc_deformation_tensor

    """
    X, Y = 30, 1
    shape = (1, X, Y, 2)
    data_org = np.zeros(shape)

    for x in range(X):
        for y in range(Y):
            data_org[0, x, y, 0] = 0.1*x

    data_exp = np.zeros(shape + (2,))
    tile_F = np.array(((1.1, 0), (0, 1)))
    
    for x in range(shape[1]):
        for y in range(shape[2]):
            data_exp[0, x, y] = tile_F
    
    F = mc.calc_deformation_tensor(data_org, 1)

    import IPython; IPython.embed()

    assert np.allclose(data_exp, \
            mc.calc_deformation_tensor(data_org, 1))


def test_calc_deformation_tensor_shear():
    """

    Unit test for mpsmechanics/dothemaths/mechanical quantities
        -> calc_deformation_tensor

    """

    shape = (1, 40, 30, 2)
    data_org = np.zeros(shape)

    for x in range(shape[1]):
        for y in range(shape[2]):
            data_org[0, x, y, 1] = 0.1*x

    data_exp = np.zeros(shape + (2,))
    tile_F = np.array(((1, 0), (0.1, 1)))
    
    for x in range(shape[1]):
        for y in range(shape[2]):
            data_exp[0, x, y] = tile_F
    
    F = mc.calc_deformation_tensor(data_org, 1)

    assert np.allclose(data_exp, \
            mc.calc_deformation_tensor(data_org, 1))

    
def test_calc_gl_strain_tensor_stretch():
    """

    Unit test for mpsmechanics/dothemaths/mechanical quantities
        -> calc_gl_strain_tensor

    """
    shape = (1, 4, 3, 2)
    data_org = np.zeros(shape)

    for x in range(shape[1]):
        for y in range(shape[2]):
            data_org[0, x, y, 0] = 0.1*x

    data_exp = np.zeros(shape + (2,))
    tile_F = np.array(((1.1, 0), (0, 1)))
    tile_C = 0.5*(np.matmul(tile_F, tile_F.T) - np.eye(2))

    for x in range(shape[1]):
        for y in range(shape[2]):
            data_exp[0, x, y] = tile_C

    assert np.allclose(data_exp, \
            mc.calc_gl_strain_tensor(data_org, 1))


def test_calc_gl_strain_tensor_shear():
    """

    Unit test for mpsmechanics/dothemaths/mechanical quantities
        -> calc_gl_strain_tensor

    """
    shape = (1, 4, 3, 2)
    data_org = np.zeros(shape)

    for x in range(shape[1]):
        for y in range(shape[2]):
            data_org[0, x, y, 1] = 0.1*x

    data_exp = np.zeros(shape + (2,))
    tile_F = np.array(((1, 0), (0.1, 1)))
    tile_C = 0.5*(np.matmul(tile_F, tile_F.T) - np.eye(2))

    for x in range(shape[1]):
        for y in range(shape[2]):
            data_exp[0, x, y] = tile_C

    assert np.allclose(data_exp, \
            mc.calc_gl_strain_tensor(data_org, 1))


def test_calc_principal_strain_stretch():
    """

    Unit test for mpsmechanics/dothemaths/mechanical quantities
        -> calc_gl_strain_tensor

    """
    shape = (1, 4, 3, 2)
    data_org = np.zeros(shape)

    for x in range(shape[1]):
        for y in range(shape[2]):
            data_org[0, x, y, 0] = 0.1*x

    data_exp = np.zeros(shape)
    
    tile_F = np.array(((1.1, 0), (0, 1)))
    tile_C = 0.5*(np.matmul(tile_F, tile_F.T) - np.eye(2))
    [lambda1, _], [ev1, _] = np.linalg.eig(tile_C)

    for x in range(shape[1]):
        for y in range(shape[2]):
            data_exp[0, x, y] = lambda1*ev1

    assert np.allclose(data_exp, \
            mc.calc_principal_strain(data_org, 1))

    
def test_calc_principal_strain_shear():
    """

    Unit test for mpsmechanics/dothemaths/mechanical quantities
        -> calc_gl_strain_tensor

    """
    shape = (1, 4, 3, 2)
    data_org = np.zeros(shape)

    for x in range(shape[1]):
        for y in range(shape[2]):
            data_org[0, x, y, 1] = 0.1*x

    data_exp = np.zeros(shape)
    
    tile_F = np.array(((1, 0), (0.1, 1)))
    tile_C = 0.5*(np.matmul(tile_F, tile_F.T) - np.eye(2))
    [lambda1, _], [ev1, _] = np.linalg.eig(tile_C)

    for x in range(shape[1]):
        for y in range(shape[2]):
            data_exp[0, x, y] = lambda1*ev1

    assert np.allclose(data_exp, \
            mc.calc_principal_strain(data_org, 1))


if __name__ == "__main__":
    test_calc_deformation_tensor_stretch()
    test_calc_gl_strain_tensor_stretch()
    test_calc_principal_strain_stretch()
    test_calc_deformation_tensor_shear()
    test_calc_gl_strain_tensor_shear()
    test_calc_principal_strain_shear()
