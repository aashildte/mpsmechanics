"""

Åshild Telle / Simula Research Laboratory / 2019

"""

import os
import numpy as np

from .folder_structure import get_input_properties

def read_prev_layer(input_file, layer_fn, param_list = [{}], overwrite=False):
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

    filename = generate_filename(input_file, layer_fn.__name__, \
            param_list, ".npy")

    print('Looking for file: ', filename)
    print("overwrite", overwrite)
    print("is file: ", os.path.isfile(filename))
    print("not overwrite & is file: ", (not overwrite and os.path.isfile(filename)))

    if not overwrite and os.path.isfile(filename):
        return np.load(filename, allow_pickle=True).item()

    return layer_fn(input_file, \
                    overwrite=overwrite, \
                    overwrite_all=overwrite, \
                    param_list=param_list)


def get_full_filename(input_file, filename, subfolder=""):
    """

    Gives filename in which the data will be stored; generates
    default structure for saving data.

    Args:
        input_file - BF/Cyan/Red file; original data
        filename - filename in which the new data will be saved
        subfolder - if it is to be saved in a subfolder (applicable
            for plots and animations)

    Returns:
        path from where input file is saved and onwards 

    """
    path, infix, _ = get_input_properties(input_file)
    output_path = os.path.join(path, infix, "mpsmechanics", subfolder)
    os.makedirs(output_path, exist_ok=True)
    
    return os.path.join(output_path, filename)


def generate_filename(input_file, script_name, param_list, extention, subfolder=""):
    """

    Generates filename based on given input parameters.

    Args:
        input_file
        script - which script it's called from
        param_list - list of parameters which the script (directly
            or through dependencies) depends on
        extention - what kind of file (e.g. "npy", "png")
        subfolder - if it is to be saved in a subfolder (applicable
            for plots and animations)


    Returns:
        filename - input file/mpsmechanics/{script_name}_{parameters}

    """

    param_name = script_name

    for param_dict in param_list:
        key_list = list(param_dict.keys())
        key_list.sort()

        for key in key_list:
            value = param_dict[key]
            param_name += f"__{key}_{value}"
    
    param_name = param_name.replace(".", "p") + extention

    return get_full_filename(input_file, param_name, subfolder=subfolder)


def save_dictionary(filename, dictionary):
    """

    Function to saving data for a specific given layer.

    Args:
        input_file - filename, including path, of original file
        layer - implies subfolder
        dictionary - values to save

    """

    #TODO do we need to make dirs here?

    output_path = os.path.split(filename)[0]
    os.makedirs(output_path, exist_ok=True)

    np.save(filename, dictionary)

    print(f"Values saved in {filename}.")
