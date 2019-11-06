"""

Åshild Telle / Simula Research Laboratory / 2019

"""

from . import dothemaths
from . import fibre_direction
from . import mechanical_analysis
from . import motion_tracking
from . import pillar_tracking
from . import visualization
from . import statistical_analysis

from .dothemaths.angular import (
    calc_projection,
    calc_projection_fraction,
    calc_angle_diff,
    flip_values,
)

from .dothemaths.heartbeat import (
        calc_beat_maxima,
        calc_beat_intervals,
        calc_beatrate,
)

from .dothemaths.operations import (
        calc_magnitude,
        calc_norm_over_time,
)

from .dothemaths.mechanical_quantities import (
    calc_deformation_tensor,
    calc_gl_strain_tensor,
    calc_principal_strain,
)
from .mechanical_analysis.mechanical_analysis import (
        analyze_mechanics,
)
from .motion_tracking.motion_tracking import (
        track_motion,
        MotionTracking,
)

from .motion_tracking.restore_resolution import refine

from .pillar_tracking.pillar_tracking import (
        track_pillars,
        track_pillars_sgvalue,
)

from .statistical_analysis.statistical_analysis import (
        calculate_stats_chips,
)
from .statistical_analysis.metrics import (
        calculate_metrics,
        calculate_metrics_all,
)

from .visualization.overtime import (
        visualize_over_time,
)
from .visualization.mechanics_spatial import (
    visualize_mechanics_spatial,
    animate_vectorfield,
    plot_vectorfield_at_peak,
    animate_decomposition,
    plot_decomposition_at_peak,
)

from .visualization.distributions import (
    visualize_distributions,
)

from .visualization.calcium_spatial import (
        visualize_calcium_spatial,
)

from .utils.iofuns.command_line import (
        get_input_files,
        add_default_parser_arguments,
        add_animation_parser_arguments,
)

from .utils.iofuns.data_layer import (
        read_prev_layer,
        get_full_filename,
        save_dictionary,
)
