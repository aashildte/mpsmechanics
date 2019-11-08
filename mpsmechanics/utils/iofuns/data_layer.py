"""



Åshild Telle / Simula Research Laboratory / 2019

"""

import os
import numpy as np

from .folder_structure import get_input_properties

def read_prev_layer(input_file, layer, layer_fn, save_data=True):
    """

    Reads data from a layer "up" in the hierarchy. If already
    calculated, we'll use previous calculations. If not, the
    relevant functions are called, and the calculated data can
    be saved for possibly reusability.

    Args:
        filename - nd2 file
        layer - string indicating which layer is needed – the program
            will look for a npy file with this prefix
        layer_fn - function to be called if data not avaiable
        save_data - default True; can be set to False

    Returns:
        dictionary with calculated values

    """
    
    path, filename, ext = get_input_properties(input_file)

    assert (ext == "nd2" or ext == "zip"), \
            "File must be an nd2 or zip file"

    folder = os.path.join(path, filename, "mpsmechanics")
    data_path = os.path.join(folder, layer + ".npy")

    print('Looking for file: ', data_path)
    
    if not os.path.isfile(data_path):
        print("Previous data not accessible. Recalculating ...")
        return layer_fn(input_file, save_data=save_data)

    print("Previous data found, loading ...")
    return np.load(data_path, allow_pickle=True).item()


def get_full_filename(input_file, layer):
    """

    Gives filename in which the data will be stored when
    save_dictionary is called, if applicable.

    """
    path, filename, _ = get_input_properties(input_file)
    output_path = os.path.join(path, filename, "mpsmechanics")
    return os.path.join(output_path, layer + ".npy")


def save_dictionary(input_file, layer, dictionary):
    """

    Function to saving data for a specific given layer.

    Args:
        input_file - filename, including path, of original file
        layer - implies subfolder
        dictionary - values to save

    """
    
    path, filename, _ = get_input_properties(input_file)
    output_path = os.path.join(path, filename, "mpsmechanics")
    os.makedirs(output_path, exist_ok=True)

    output_file = os.path.join(output_path, layer + ".npy")
    
    np.save(output_file, dictionary)

    print(f"Values saved in {output_file}.")

