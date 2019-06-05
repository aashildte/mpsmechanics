"""

Åshild Telle / Simula Research Labratory / 2019

Default parameters; global range to be aviable for all scripts,
such that they are directly comparable.

"""


def get_default_parameters():

    alpha = 0.75
    N_d = 5
    dt = 1./100                          # cl arguments? fps
    threshold_prev = 2                   # um/s
    threshold_mv = 1E-10                 # [unit]/s

    return alpha, N_d, dt, threshold_prev, threshold_mv
