"""

Åshild Telle / Simula Research Laboratory / 2019

"""

import numpy as np

import mpsmechanics as mc


def test_calc_avg_tf_filter():
    """

    Unit test for mpsmechanics/mechanical_analysis/filters
        -> calc_avg_tf_filter

    """
    shape = (1, 3, 4)
    data_org = np.zeros(shape)
    data_org[0, 1, :] += 1

    tf_filter = data_org.astype(bool)
    
    data_exp = 1

    assert np.allclose(data_exp, mc.calc_avg_tf_filter(data_org, tf_filter))


def test_calc_std_tf_filter():
    """

    Unit test for mpsmechanics/mechanical_analysis/filters
        -> calc_std_tf_filter

    """
    shape = (1, 3, 4)
    data_org = np.zeros(shape)
    data_org[0, 1, :] += 1

    tf_filter = data_org.astype(bool)
    
    data_exp = 0

    assert np.allclose(data_exp, mc.calc_std_tf_filter(data_org, tf_filter))



def test_filter_time_dependent():
    """

    Unit test for mpsmechanics/mechanical_analysis/filters
        -> filter_time_dependent

    """
    shape = (2, 3, 4, 2)
    data_org = np.zeros(shape)

    data_org[0, 1, :, :] += 1
    data_org[1, 0, :, :] += 1
    data_org[1, 2, :, :] += 1

    data_exp = data_org[:, :, :, 0].astype(bool)

    assert np.array_equal(data_exp, mc.filter_time_dependent(data_org))


def test_filter_constrained():
    """

    Unit test for mpsmechanics/mechanical_analysis/filters
        -> filter_constrained

    """
    shape = (2, 3, 4, 2)
    data_org = np.zeros(shape)

    data_org[0, 1, 1:, :] += 1
    data_org[1, 0, 1:, :] += 1
    data_org[1, 2, 1:, :] += 1

    data_exp = np.zeros(shape[:3]).astype(bool)
    data_exp[:, :, 2:] += True

    assert np.array_equal(data_exp, mc.filter_constrained(data_org, 1))


def test_filter_uniform():
    """

    Unit test for mpsmechanics/mechanical_analysis/filters
        -> filter_uniform

    """
    shape = (2, 3, 4, 2)
    data_org = np.zeros(shape)

    data_org[0, 1, :, :] += 1
    data_org[1, 0, :, :] += 1
    data_org[1, 2, :, :] += 1

    data_exp = np.full(shape[:3], True, dtype=bool)

    assert np.array_equal(data_exp, mc.filter_uniform(data_org))

if __name__ == "__main__":
    test_calc_avg_tf_filter()
    test_calc_std_tf_filter()
    test_filter_constrained()
    test_filter_time_dependent()
    test_filter_uniform()
