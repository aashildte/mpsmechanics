"""

Åshild Telle / Simula Research Laboratory / 2019

"""

import os
import numpy as np

from .folder_structure import get_input_properties, make_dir_structure

def save_dictionary(input_file, layer, dictionary):
    """

    Function to saving data for a specific given layer.

    Args:
        input_file - filename, including path, of original file
        layer - implies subfolder
        dictionary - values to save

    """
    
    path, filename, _ = get_input_properties(input_file)
    output_path = os.path.join(path, filename)
    make_dir_structure(output_path)

    output_file = os.path.join(output_path, layer + ".npy")

    np.save(output_file, dictionary)
