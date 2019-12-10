"""

Åshild Telle / Simula Research Laboratory / 2019

"""

import os
from matplotlib import animation
import matplotlib.pyplot as plt


def get_animation_configuration(params, mps_data):
    """

    Creates dictionary with standard animation properties.

    Args:
        params - parameter from command line, including
            animate (bool) and scaling_factor (float)
        mps_data - MPS object

    Return:
        Dictionary with configuration properties.

    """

    return {"animate" : params["animate"],
            "framerate" : mps_data.framerate*params["scaling_factor"],
            "num_frames" : mps_data.frames.shape[-1],
            "extension" : "mp4"}


def make_animation(fig, update, fname, num_frames, framerate, extension):

    extensions = ["gif", "mp4"]
    msg = "Invalid extension {}. Expected one of {}".format(extension, extensions)
    assert extension in extensions, msg

    # Set up formatting for the movie files
    if extension == "mp4":
        writer = animation.writers["ffmpeg"](fps=framerate)
    else:
        writer = animation.writers["imagemagick"](fps=framerate)

    anim = animation.FuncAnimation(fig, update, num_frames)

    fname = os.path.splitext(fname)[0]
    anim.save("{}.{}".format(fname, extension), writer=writer)
    plt.close('all')
