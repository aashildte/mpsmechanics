"""

Åshild Telle / Simula Research Laboratory / 2019

"""

from .command_line import get_input_files, split_parameter_dictionary


def run_script(function, description, parser, channel, keys):
    """

    Runs a script in a "default" way, which should work across all levels.

    """

    vargs = vars(parser.parse_args())

    input_files = get_input_files(vargs.pop("input_files"), channel)
    debug_mode = vargs.pop("debug_mode")
    overwrite = vargs.pop("overwrite")
    overwrite_all = vargs.pop("overwrite_all")

    param_list = split_parameter_dictionary(vargs, keys)

    for f_in in input_files:
        if debug_mode:
            function(f_in, overwrite, overwrite_all, param_list)
        else:
            try:
                function(f_in, overwrite, overwrite_all, param_list)
            except Exception as exp:
                print(
                    f"Could not run {description}; error msg: {exp}"
                )
                if f_in is not input_files[-1]:
                    print("Launching script for next file ...")
