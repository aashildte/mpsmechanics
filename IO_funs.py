"""

Module for IO operations

Åshild Telle / Simula Research Labratory / 2019

"""


import sys
import os
import numpy as np
import matplotlib.pyplot as plt


class IO_funs:
    def read_disp_file(self, filename):
        """

        Reads the input file, where the file is assumed to be a csv file on
        the form
            T, X, Y
            x0, y0
            x1, y1
            ...

        where we have T x X x Y values, giving the position for a given unit
        for time step t0, t1, ..., for  x coordinates x0, x1, ... and
        y coordinates y0, y1, ...

        x0, y0 gives relative motion of unit (0, 0) at time step 0; x1, y1
        gives relative motion of unit (1, 0) at time step 0, etc.

        Arguments:
            filename - csv file

        Returns:
            4-dimensional numpy array, of dimensions T x X x Y x 2

        """

        f = open(filename, 'r')

        T, X, Y = map(int, str.split(f.readline(), ","))

        data = np.zeros((T, X, Y, 2))

        for t in range(T):
            for i in range(X):
                for j in range(Y):
                    d = str.split(f.readline().strip(), ",")
                    data[t, i, j] = np.array(d)

        f.close()

        return data

    def make_dir_structure(self, path):
        """

        Makes a directory structure based on a given path.

        """

        # Folder structure different depending on OS,
        # check and assign different for Windows and Linux/Mac

        os_del = "\\" if os.name=="nt" else "/"

        dirs = path.split(os_del)

        acc_d = "."

        for d in dirs:
            acc_d = acc_d + os_del + d
            if not (os.path.exists(acc_d)):
                os.mkdir(acc_d)


if __name__ == "__main__":
    
    try:
        f_in = sys.argv[1]
    except:
        print("Give file name as first argument")
        exit(-1)

    IO = IO_funs()

    data = IO.read_disp_file(f_in)

    assert(len(data.shape)==4)

    print("Read data test passed")
    print("All tests passed")
