"""



Åshild Telle / Simula Research Laboratory / 2019

"""

import os
import numpy as np

from .folder_structure import get_input_properties

def read_prev_layer(input_file, layer, layer_fn, save_data=True):
    """

    Reads data from a layer "up" in the hierarchy. If already
    calcualted, we'll use previous calculations. If not, the
    relevant functions are called, and the calculated data can
    be saved for possibly reusability.

    Args:
        filename - nd2 file
        layer - string indicating which layer is needed
        layer_fn - function to be called if data not avaiable
        save_data - default True; can be set to False

    Returns:
        dictionary with calculated values

    """
    
    path, filename, ext = get_input_properties(input_file)

    assert ext == "nd2", "File must be an nd2 file"

    data_path = os.path.join(path, \
            os.path.join(filename, layer + ".npy"))
    
    if not os.path.isfile(data_path):
        print("Previous data not accessible. Recalculating ..")
        return layer_fn(input_file, save_data=save_data)

    print("Previous data found, loading ..")
    return np.load(data_path, allow_pickle=True).item()
