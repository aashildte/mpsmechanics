"""

Åshild Telle / Simula Research Laboratory / 2019

"""

import numpy as np

import mpsmechanics as mc

def test_calc_norm_over_time():
    """

    Unit test for mpsmechanics/dothemaths/operations
        -> calc_norm_over_time

    """

    shape = (5, 4, 3, 2)

    data_org = np.zeros(shape)
    data_exp = np.zeros(shape[0])

    for t in range(shape[0]):
        data_org[t] = (0.1*t)*np.ones(shape[1:])
        data_exp[t] = np.sqrt(2*(0.1*t)**2)*shape[1]*shape[2]

    assert np.allclose(data_exp,
            mc.calc_norm_over_time(data_org))


def test_calc_magnitude():
    """

    Unit test for mpsmechanics/dothemaths/operations
        -> calc_magnitude

    """

    shape = (5, 4, 3, 2)

    data_org = np.zeros(shape)
    data_exp = np.zeros(shape[:3])

    for t in range(shape[0]):
        data_org[t] = (0.1*t)*np.ones(shape[1:])
        data_exp[t] = np.sqrt(2*(0.1*t)**2)*np.ones(shape[1:-1])
   
    assert np.allclose(data_exp,
            mc.calc_magnitude(data_org))


def test_normalize_values():
    """

    Unit test for mpsmechanics/dothemaths/operations
        -> normalize_values

    """

    shape = (5, 4, 3, 2)

    data_org = np.zeros(shape)
    data_exp = 1/np.sqrt(2)*np.ones(shape)
    data_exp[0] *= 0

    for t in range(shape[0]):
        data_org[t] = (0.1*t)*np.ones(shape[1:])
  
    assert np.allclose(data_exp,
            mc.normalize_values(data_org))


if __name__ == "__main__": 
    test_calc_magnitude()
    test_calc_norm_over_time()
    test_normalize_values()
