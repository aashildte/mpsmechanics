"""

Given a list of input files on the command line (could be all files
in a given directory), where each file is either a csv or a nd2 file
containing displacement data (ref. README). 

Run as
    python plot_disp_strain.py [files]

The output (colour maps) are saved in Plot_disp_str, being a subfolder
of Figures.

"""


import os
import sys
import numpy as np
import matplotlib.colors as cl

import dothemaths.operations as op
import dothemaths.preprocessing as pp
import dothemaths.mechanical_properties as mc
import dothemaths.plot_vector_field as pl
import iofuns.io_funs as io

def read_values(f_in):
    """

    Given the displacement data, this function finds the time slot 
    with most detecteddisplacement and extracts the relevant data +
    principal strain for the same time step.

    Arguments:
        data - T x X x Y x 2 numpy array 

    Returns:
        displacement data - X x Y x 2 numpy array
        principal strain - X x Y x 2 numpy array

    """

    xlen = 664.30

    disp_data, scale = io.read_file(f_in, xlen)
    disp_data = pp.do_diffusion(disp_data, 0.75, 5)
    
    time_step = op.calc_max_ind(op.calc_norm_over_time(disp_data))
    disp_t = disp_data[time_step]

    strain_t = mc.calc_principal_strain(disp_t, over_time=False)

    return scale*disp_t, strain_t


def gather_data(filenames):
    """
    
    Given a list of filenames, this function extracts displacement
    and principal strain for relevant time steps (at maxima).

    Arguments:
        filenames - list of filenames

    Returns:
        displacement data - list of numpy arrays
        principal strain - list of numpy arrays

    """

    displacement_data = []
    principal_strain = []

    for f_in in sys.argv[1:]:
        disp, strain = read_values(f_in)

        displacement_data.append(disp)
        principal_strain.append(strain)

    return displacement_data, principal_strain


def plot_all_values(disp, strain, idts, max_disp, max_strain):

    # colour maps

    norm1 = cl.Normalize(0.0, max_disp)
    norm2 = cl.LogNorm(.1, max_strain + .1)  # positive values only
    #norm2 = cl.Normalize(0.0, max_strain)

    # colorbars - here or combined with every plot?
    #cb1 = mpl.colorbar.ColorbarBase(ax,
    #                                norm=norm1,
    #                                orientation='horizontal')
    #cb1.set_label('Displacement')
    #plt.savefig(path + de + "Displacement.png", dpi=1000)

    # folder structure

    path = os.path.join("Figures", "Plot_disp_strain")
    io.make_dir_structure(path)

    norms = [norm1, norm2]

    dimensions = (664.30, 381.55)
    labels = ["Displacement", "Principal strain"]

    for (disp, strain, idt) in \
            zip(displacement_data, principal_strain, idts):

        # filter values by 0 displacement (?)

        #X, Y = disp.shape[:2]
        #for x in range(X):
        #    for y in range(Y):
        #        if (np.linalg.norm(disp[x, y]) < 1E-10):
        #            strain[x, y] = np.array([1E-14, 1E-14])

        vector_fields = [disp, strain]

        pl.plot_direction_and_magnitude(vector_fields, norms, labels, \
                dimensions, path, idt)


try:
    assert(len(sys.argv)>1)
except:
    print("Give file names as positional arguments.")
    exit(-1)

# gather (todo: can do in parallel)

displacement_data, principal_strain = gather_data(sys.argv[1:])
idts = [io.get_idt(f_in) for f_in in sys.argv[1:]]

# maxima (syncronization step)

max_disp = max([np.max(d) for d in displacement_data])
max_strain = max([np.max(p) for p in principal_strain])

# plotting (can again be done in parallel)

plot_all_values(displacement_data, principal_strain, idts, \
        max_disp, max_strain)
