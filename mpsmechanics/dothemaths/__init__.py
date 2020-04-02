"""

Åshild Telle / Simula Research Laboratory / 2019

"""


from .angular import (
    calc_projection,
    calc_projection_fraction,
    flip_values,
)

from .heartbeat import (
    calc_beat_maxima,
    calc_beat_intervals,
    calc_beatrate,
)

from .interpolation import interpolate_values_xy

from .operations import (
    calc_magnitude,
    calc_norm_over_time,
    normalize_values,
)

from .mechanical_quantities import (
    calc_gradients,
    calc_deformation_tensor,
    calc_gl_strain_tensor,
    calc_principal_strain_vectors,
    calc_principal_strain_scalars,
)
