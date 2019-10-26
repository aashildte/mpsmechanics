
import numpy as np
import mpsmechanics as mc


def test_calc_deformation_tensor():
    """

    Unit test for mpsmechanics/dothemaths/mechanical quantities
        -> calc_deformation_tensor

    """

    shape = (2, 4, 3, 2)
    data_org = np.zeros(shape)

    for x in range(shape[1]):
        for y in range(shape[2]):
            data_org[1, x, y, 0] = 0.2*x
            data_org[1, x, y, 1] = 0.1*y
    

    data_exp = np.zeros(shape + (2,))    
    tile_t0 = np.eye(2)
    tile_t1 = np.eye(2); tile_t1[0][0] = 1.2; tile_t1[1][1] = 1.1

    for x in range(shape[1]):
        for y in range(shape[2]):
            data_exp[0, x, y] = tile_t0
            data_exp[1, x, y] = tile_t1

    assert np.allclose(data_exp, 
                       mc.calc_deformation_tensor(data_org, 1))


if __name__ == "__main__":
    test_calc_deformation_tensor()
