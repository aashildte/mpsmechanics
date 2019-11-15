"""

Åshild Telle / Simula Research Laboratory / 2019

"""


import numpy as np
import matplotlib as plt
import mps
import mpsmechanics as mm
import glob
import os
import sys
from argparse import ArgumentParser

def visualize_calcium_spatial(input_file, animate=False, overwrite=False, \
        magnitude_scale="linear", framerate_scale=0.2):
    
    data = mps.MPS(input_file)    
    N = data.size_x

    output_folder = os.path.join(input_file[:-4], "plots_spatial")
    fout = os.path.join(output_folder, f"corrected_{N}.npy")
    
    if (not overwrite) and os.path.isfile(fout):
        return
    
    os.makedirs(output_folder, exist_ok=True)

    avg = mps.analysis.local_averages(data.frames, data.time_stamps, N=N)
    np.save(fout, avg)

    print(f"Local averages found for {input_file}; making the plots ..")

    avg = np.swapaxes(np.swapaxes(avg, 0, 1), 0, 2)
    avg = avg[:,:,:,None]
    
    pixels2um = data.info["um_per_pixel"]
    label = "Pixel intensity, relative to baseline"
    fname = os.path.join(output_folder, "original_vs_corrected")
    mm.plot_decomposition_at_peak(avg, magnitude_scale, label, pixels2um, data.frames, fname)

    if animate:
        print("Making a movie ...")
        mm.animate_decomposition(avg, magnitude_scale, label, pixels2um, \
             data.frames, fname, framerate=framerate_scale*data.framerate)
