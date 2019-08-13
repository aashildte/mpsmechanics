
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import animation

import mps

from ..utils.iofuns.data_layer import read_prev_layer
from ..utils.iofuns.folder_structure import make_dir_structure, \
        make_dir_layer_structure
from ..mechanical_analysis.mechanical_analysis import analyze_mechanics
from ..pillar_tracking.pillar_tracking import track_pillars

def animate_vectorfield(vectors, fname="animation", framerate=None, images=None, \
            extension="mp4", dpi=300, dx=5):

    D = vectors.shape[-1]

    if D!=2:
        print("Only defined for 2D vectors")
        return

    # from motion tracking

    extensions = ["gif", "mp4"]
    msg = f"Invalid extension {extension}. Expected one of {extensions}"
    assert extension in extension, msg
    
    if images is not None:
        Nx, Ny = images.shape[1:3]
    else:
        Nx, Ny = vectors.shape[1:3]

    x = np.linspace(0, Nx, vectors.shape[1])
    y = np.linspace(0, Ny, vectors.shape[2])


    block_size = Nx // vectors.shape[1]
    scale = (np.max(vectors) - np.min(vectors))  
    print("scale: ", scale)

    figsize = (2 * Ny / vectors.shape[2], 2 * Nx / vectors.shape[1])
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    if images is not None:
        im = ax.imshow(images[:, :, 0], cmap=cm.gray)

    Q = ax.quiver(
        y[::dx],
        x[::dx],
        -vectors[0, ::dx, ::dx, 1],
        vectors[0, ::dx, ::dx, 0],
        color="r",
        units="xy",
        scale_units="inches",
        scale=scale,
    )
    ax.set_aspect("equal")

    def update(idx):
        Q.set_UVC(-vectors[idx, ::dx, ::dx, 1], vectors[idx, ::dx, ::dx, 0])
        
        if images is not None:
            im.set_array(images[:, :, idx])

    # Set up formatting for the movie files
    if extension == "mp4":
        Writer = animation.writers["ffmpeg"]
    else:
        Writer = animation.writers["imagemagick"]
    writer = Writer(fps=framerate)

    N = vectors.shape[0]
    anim = animation.FuncAnimation(fig, update, N)

    fname = os.path.splitext(fname)[0]
    anim.save(f"{fname}.{extension}", writer=writer)


def visualize_vectorfield(f_in, layers, save_data=True):
    """

    Visualize fields - "main function"

    """
    layers = layers.split(" ")

    mt_data = mps.MPS(f_in)

    for layer in layers:
        layer_fn = eval(layer)
    
        output_folder = os.path.join(\
                make_dir_layer_structure(f_in, \
                "visualize_vectorfield"), layer)
        make_dir_structure(output_folder)

        data = read_prev_layer(f_in, layer, layer_fn, save_data)

        # average over time
         
        for key in data["all_values"].keys(): 
            animate_vectorfield(data["all_values"][key], \
                    framerate=0.1*mt_data.framerate, \
                    fname=os.path.join(output_folder, "vectorfield_" + key))

