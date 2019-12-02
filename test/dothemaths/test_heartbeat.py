"""

Åshild Telle / Simula Research Laboratory / 2019

"""

import numpy as np

import mpsmechanics as mc

def test_calc_beat_maxima():
    """

    Unit test for mpsmechanics/dothemaths/heartbeat
        -> calc_beat_maxima

    """

    data_org = np.tile(np.array((0, 0, 1, 2, 1, 0, 0)), 4)
    data_exp = np.array([(3 + 7*i) for i in range(4)])

    assert np.allclose(data_exp,
                       mc.calc_beat_maxima(data_org,
                                           disp_threshold=1))


def test_calc_beat_intervals():
    """

    Unit test for mpsmechanics/dothemaths/heartbeat
        -> calc_beat_maxima

    """
    data_org = np.tile(np.array((0, 0, 1, 2, 1, 0, 0)), 4)
    data_exp = [(6 + 7*i, 6+7*(i+1)) for i in range(3)]

    assert np.allclose(data_exp,
                       mc.calc_beat_intervals(data_org,
                                              disp_threshold=1))


def test_calc_beatrate():
    """

    Unit test for mpsmechanics/dothemaths/heartbeat
        -> calc_beatrate

    """
    shape = (7, 4, 3)
    data_org = np.zeros(shape)
    maxima = [1, 3, 5]
    intervals = [(0, 2), (2, 4)]

    for i in range(1, shape[0], 2):
        data_org[i] = np.ones(shape[1:])

    data_exp_spatial = 500*np.ones((2, 4, 3))

    assert np.allclose(data_exp_spatial, 
                       mc.calc_beatrate(data_org, maxima, \
                                        intervals, np.arange(shape[0])))

if __name__ == "__main__":
    test_calc_beat_maxima()
    test_calc_beat_intervals()
    test_calc_beatrate()
