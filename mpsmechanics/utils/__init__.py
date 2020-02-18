"""

Åshild Telle / Simula Research Laboratory / 2019

"""


from .folder_structure import (
    get_input_properties,
    make_dir_layer_structure,
)

from .command_line import (
    add_animation_parser_arguments,
    add_default_parser_arguments,
    add_focus_parser_arguments,
    add_parameters_parser_arguments,
    get_input_files,
    split_parameter_dictionary,
)

from .run_script import (
    run_script
)

from .data_layer import (
    generate_filename,
    get_full_filename,
    read_prev_layer,
    save_dictionary,
    write2read,
)

