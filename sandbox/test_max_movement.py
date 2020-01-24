import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import cm


from mpsmechanics.motion_tracking import MotionTracking


class MPSData:
    def __init__(self, data):
        self.frames = data.get("frames")
        self.time_stamps = data.get("time")
        self.framerate = data.get("framerate")
        self.info = data.get("info")
        self.num_frames = self.info.get("num_frames")


def animate_vectorfield(vectors, data, fname="animation", extension="mp4"):

    images = data.frames

    extensions = ["gif", "mp4"]
    msg = f"Invalid extension {extension}. Expected one of {extensions}"
    assert extension in extension, msg
    if images is not None:
        Nx, Ny = images.shape[:2]
    else:
        Nx, Ny = vectors.shape[:2]

    x = np.linspace(0, Nx, vectors.shape[0])
    y = np.linspace(0, Ny, vectors.shape[1])

    block_size = Nx // vectors.shape[0]
    scale = max(np.divide(vectors.shape[:2], block_size)) * 2
    dx = 1

    figsize = (2 * Ny / vectors.shape[1], 2 * Nx / vectors.shape[0])
    fig, ax = plt.subplots(figsize=figsize)
    if images is not None:
        im = ax.imshow(images[:, :, 0], cmap=cm.gray)
    Q = ax.quiver(
        y[::dx],
        x[::dx],
        -vectors[::dx, ::dx, 1, 0],
        vectors[::dx, ::dx, 0, 0],
        color="r",
        units="xy",
        scale_units="inches",
        scale=scale,
    )
    ax.set_aspect("equal")

    def update(idx):
        Q.set_UVC(-vectors[::dx, ::dx, 1, idx], vectors[::dx, ::dx, 0, idx])
        if images is not None:
            im.set_array(images[:, :, idx])

    # Set up formatting for the movie files
    if extension == "mp4":
        Writer = animation.writers["ffmpeg"]
    else:
        Writer = animation.writers["imagemagick"]
    writer = Writer(fps=data.framerate)

    N = vectors.shape[-1]
    anim = animation.FuncAnimation(fig, update, N)

    fname = os.path.splitext(fname)[0]
    anim.save(f"{fname}.{extension}", writer=writer)


def generate_test_data(fname):
    """Take out a little part that is faster to process
    """
    import mps

    d = mps.MPS("PointMM_4_ChannelBF_VC_Seq0000.nd2")
    q = d.frames[100:140, 100:140, :]
    d.info["size_x"] = q.shape[0]
    d.info["size_y"] = q.shape[1]
    data = {
        "frames": q,
        "time": d.time_stamps,
        "framerate": d.framerate,
        "info": d.info,
        "num_frames": q.shape[-1],
    }
    np.save(fname, data)


def process(all_vectors, outfolder, data, mbs=None):
    if mbs is None:
        mbs = range(all_vectors.shape[0])
    # all_vector mb x X x Y x coord x T
    # Take the norm and collect the maximum displacement over time
    q = np.linalg.norm(np.array(all_vectors), axis=(3,)).max(-1)

    # Plot the maximum computed displacement for each max_block_movement
    # We would except this graph to be like a roofline, i.e
    # first linear increase and then constant
    fig, ax = plt.subplots()
    for i in range(q.shape[1]):
        for j in range(q.shape[2]):
            ax.plot(mbs, q[:, i, j])
    ax.set_xlabel("max allowed movement")
    ax.set_ylabel("max computed displacement")
    fig.savefig(os.path.join(outfolder, "max_move_disp"))

    # For each max_block_movement visualize the maximum displacement
    # for each block
    fig, ax = plt.subplots(5, 3, figsize=(12, 10))
    for i, axi in enumerate(ax.flatten()):
        axi.set_title(i)
        m = axi.imshow(q[i, :, :])
        fig.colorbar(m, ax=axi)
    fig.savefig(os.path.join(outfolder, "max_final_mb"))

    # Do the same as in the previous figure but make all of them
    # the same scale for the colormap
    vmax = q.max()
    fig, ax = plt.subplots(5, 3, figsize=(12, 10))
    for i, axi in enumerate(ax.flatten()):
        axi.set_title(i)
        m = axi.imshow(q[i, :, :], vmin=0, vmax=vmax)
        fig.colorbar(m, ax=axi)
    fig.savefig(os.path.join(outfolder, "max_final_mb_same_scale"))

    # Investigate final mb and find the block index with the
    # largest displacement and plot this as a funciton of time
    q_final = q[-1, :, :]
    idx = np.unravel_index(q_final.argmax(), q_final.shape)
    w = all_vectors[-1, idx[0], idx[1], :, :]
    wx = w[0]
    wy = w[1]
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(data.time_stamps, wx)
    ax[0].set_title("displacement x")
    ax[1].plot(data.time_stamps, wy)
    ax[1].set_title("displacement y")
    fig.savefig(os.path.join(outfolder, "high_displacement_point"))
    plt.close("all")


def main():

    # Make a smaller data set
    test_data_file = "test_data.npy"
    if not os.path.isfile(test_data_file):
        generate_test_data(test_data_file)
    data = MPSData(np.load(test_data_file, allow_pickle=True).item())

    motion = MotionTracking(
        data, block_size=3, max_block_movement=3, filter_kernel_size=8
    )

    vectors = motion.displacement_vectors
    exit()
    mbs = range(15)

    for filter_kernel_size in [4, 8, 12]:
        outfolder = f"output_fks{filter_kernel_size}"
        os.makedirs(outfolder, exist_ok=True)
        # Loop over different values of max_block_movement
        # Compute the displacement and save all the vectors to a file for faster runs later
        all_vectors_file = os.path.join(outfolder, "all_vectors.npy")
        if os.path.isfile(all_vectors_file):
            all_vectors = np.load(all_vectors_file)
        else:
            all_vectors = []
            for mb in mbs:
                motion = MotionTracking(
                    data,
                    block_size=3,
                    max_block_movement=mb,
                    filter_kernel_size=filter_kernel_size,
                )

                vectors = motion.displacement_vectors

                all_vectors.append(vectors)
                animate_vectorfield(
                    vectors, data, os.path.join(outfolder, f"displacement_mb{mb}")
                )
            all_vectors = np.array(all_vectors)
            np.save(all_vectors_file, all_vectors)

        process(all_vectors, outfolder, data, mbs)


if __name__ == "__main__":
    main()
