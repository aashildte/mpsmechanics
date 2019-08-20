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
    
    data_path = os.path.join(os.path.join(path, filename),\
                             layer + ".npy")

    print('Looking for file: ', data_path)
    
    if not os.path.isfile(data_path):
        print("Previous data not accessible. Recalculating ...")
        return layer_fn(input_file, save_data=save_data)

    print("Previous data found, loading ...")
    return np.load(data_path, allow_pickle=True).item()
