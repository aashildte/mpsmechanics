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
            mc.calc_beat_maxima(data_org, disp_threshold=1))


def test_calc_beat_intervals():
    """

    Unit test for mpsmechanics/dothemaths/heartbeat
        -> calc_beat_maxima

    """
    data_org = np.tile(np.array((0, 0, 1, 2, 1, 0, 0)), 4)
    data_exp = [(6 + 7*i, 6+7*(i+1)) for i in range(3)]

    assert np.allclose(data_exp,
            mc.calc_beat_intervals(data_org, disp_threshold=1))


def test_calc_beatrate():
    """

    Unit test for mpsmechanics/dothemaths/heartbeat
        -> calc_beatrate

    """
    shape = (5, 4, 3)
    data_org = np.zeros(shape)
    intervals = [(0, 3), (3, 5)]

    for i in range(1, shape[0], 2):
        data_org[i] = np.ones(shape[1:])

    data_exp_spatial = 500*np.ones((2, 4, 3))
    avg_exp = 500
    std_exp = 0

    beatrate_spatial, avg, std = \
            mc.calc_beatrate(data_org, intervals, \
                             np.arange(shape[0]))

    assert np.allclose(data_exp_spatial, beatrate_spatial)
    assert np.allclose(avg_exp, avg)
    assert np.allclose(std_exp, std)


if __name__ == "__main__": 
    test_calc_beat_maxima()
    test_calc_beat_intervals()
    test_calc_beatrate()
