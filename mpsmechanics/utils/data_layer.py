"""

Åshild Telle / Simula Research Laboratory / 2019

"""

import os
import numpy as np

from .folder_structure import get_input_properties

def read_prev_layer(input_file, layer_fn, param_list, overwrite):
    """

    Reads data from a layer "up" in the hierarchy. If already
    calculated, we'll use previous calculations. If not, the
    relevant functions are called, and the calculated data can
    be saved for possibly reusability.

    Args:
        filename - nd2 file
        layer_fn - function to be called if data not avaiable
        param_list
        overwrite

    Returns:
        dictionary with calculated values

    """

    path, filename, ext = get_input_properties(input_file)

    assert (ext == "nd2" or ext == "zip"), \
            "File must be an nd2 or zip file"

    folder = os.path.join(path, filename, "mpsmechanics")
    filename = generate_filename(input_file, layer_fn.__name__, param_list) + ".npy"
    data_path = os.path.join(folder, filename)

    print('Looking for file: ', data_path)

    return layer_fn(input_file, overwrite=overwrite, param_list=param_list)


def get_full_filename(input_file, layer):
    """

    Gives filename in which the data will be stored when
    save_dictionary is called, if applicable.

    """
    path, filename, _ = get_input_properties(input_file)
    output_path = os.path.join(path, filename, "mpsmechanics")
    return os.path.join(output_path, layer)


def generate_filename(input_file, script_name, param_list):
    """

    Generates filename based on given input parameters.

    Args:
        input_file
        script - which script it's called from
        param_list - list of parameters which the script (directly
            or through dependencies) depends on

    Returns:
        filename - input file/mpsmechanics/[script_name]_[parameters]

    """

    param_name = script_name

    for param_dict in param_list:
        key_list = list(param_dict.keys())
        key_list.sort()

        for key in key_list:
            value = param_dict[key]
            param_name += f"_{key}:{value}"

    param_name = param_name.replace(".", "p")

    return get_full_filename(input_file, param_name)


def save_dictionary(filename, dictionary):
    """

    Function to saving data for a specific given layer.

    Args:
        input_file - filename, including path, of original file
        layer - implies subfolder
        dictionary - values to save

    """

    output_path = os.path.split(filename)[0]
    os.makedirs(output_path, exist_ok=True)

    np.save(filename, dictionary)

    print(f"Values saved in {filename}.")
