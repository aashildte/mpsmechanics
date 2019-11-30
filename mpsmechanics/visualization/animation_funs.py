"""

Åshild Telle / Simula Research Laboratory / 2019

"""

import os
from matplotlib import animation
import matplotlib.pyplot as plt

def make_animation(fig, update, num_frames, framerate, fname, extension):
    
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
