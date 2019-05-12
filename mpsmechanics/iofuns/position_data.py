
import numpy as np


def read_pt_file(f_in):
    """

    Reads in values for pillar coordinates + radii.

    Args:
        f_in - Filename

    Returns;
        Numpy array of dimensions P x 3, P being the number of
            points; entries being x, y, radius for all points.

    """

    f = open(f_in, "r")

    lines = f.read().split("\n")[1:]

    if(len(lines[-1])==0):
        lines = lines[:-1]      # ignore last line if empty

    p_values = [[int(x) for x in line.split(",")] \
                        for line in lines]  # or float???

    # using standard radius instead of input; by choice

    r_standard = 1990/664*10       # pixel coords / length * radius
    
    # flip x and y; temporal solution due to two different
    # conventions used. TODO - use same everywhere

    for i in range(len(p_values)):
        x, y, r = p_values[i]
        p_values[i] = [y, x, r_standard]

    p_values = np.array(p_values)

    f.close()

    return p_values

