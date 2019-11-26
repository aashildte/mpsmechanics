"""

Åshild Telle / Simula Research Labratory / 2019

"""

import os

def get_input_properties(filename):
    """

    Splits file into three parts: Path, filename, extension

    Arguments:
        filename - full path including filename

    Returns:
        path - path to filename
        filename - filename in that directory
        ext - file extension

    """

    path, tail = os.path.split(filename)
    filename, ext = tail.split(".")

    return path, filename, ext


def make_dir_layer_structure(f_in, layer):
    """

    If applicable, akes a subfolder with the name given by layer, in
    a subfolder with the same name as the input file, which again is
    in the same directory as the input file itself.

    Args:
        f_in - filename, including full path
        layer - name of subfolder to be created

    Returns:
        path to folder created

    """

    path, filename, _ = get_input_properties(f_in)
    output_folder = os.path.join(path, \
            os.path.join(filename, layer))
    os.makedirs(output_folder, exist_ok=True)

    return output_folder
