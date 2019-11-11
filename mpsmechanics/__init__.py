"""

Åshild Telle / Simula Research Laboratory / 2019

"""

from . import mechanical_analysis
from . import motion_tracking
from . import pillar_tracking
from . import visualization
from . import statistical_analysis


from .dothemaths import (
    calc_beat_intervals,
    calc_beat_maxima,
    calc_beatrate,
    calc_deformation_tensor,
    calc_gl_strain_tensor,
    calc_magnitude,
    calc_norm_over_time,
    calc_principal_strain,
    calc_projection,
    calc_projection_fraction,
    flip_values,
    interpolate_values_xy,
    normalize_values,
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

from .utils.iofuns.command_line import (
        get_input_files,
        add_default_parser_arguments,
        add_animation_parser_arguments,
        add_parameters_parser_arguments,
)

from .utils.iofuns.data_layer import (
        read_prev_layer,
        get_full_filename,
        save_dictionary,
)


from .visualization import (
    animate_decomposition,
    animate_vectorfield,
    plot_decomposition_at_peak,
    plot_vectorfield_at_peak,
    visualize_calcium_spatial,
    visualize_distributions,
    visualize_mechanics,
    visualize_over_time,
    visualize_vectorfield,
)
