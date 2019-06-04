def write_position_values(values, filename):
    """

    Output to files: T x N x 2 values 

    Args:
        values - numpy array of dimension T x N x 2
        filename - save as

    """

    T, N = values.shape[:2]

    f = open(filename, "w")

    f.write(str(T) + ", " + str(N) + "\n")

    for t in range(T):
        for n in range(N):
            x, y = values[t, n]
            f.write(str(x) + ", " + str(y) + "\n")

    f.close()


def write_max_values(max_indices, output_d, filename):
    """

    Writes values at maximum displacement to a file.

    Args:
        max_indices - list-alike structure for indices of maxima
        output_d - dictionary with attributes as keys, and numbers/
            calculated values (in (x, y) pairs; N x 2 numpy array)
            as values 
        filename - save as

    """

    f = open(filename, "w")

    max_str = " ," + ", , ".join([str(m) \
            for m in max_indices]) + "\n"
    f.write(max_str)

    for k in output_d.keys():
        m_values = output_d[k]
        maxima_str = ", ".join([str(m[0]) + ", " + str(m[1]) \
                for m in m_values])

        pos_str = k + ", " + maxima_str + "\n"
        f.write(pos_str)

    f.close()


