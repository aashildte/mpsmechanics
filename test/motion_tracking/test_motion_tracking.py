import itertools
import os
from collections import namedtuple

import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import cm
from matplotlib import animation
from skimage import filters

import mps
import mpsmechanics as mc

mps.set_log_level(10)
MPSData = namedtuple(
    "MPSData", ["frames", "time_stamps", "info", "framerate", "num_frames"]
)
block_sizes = [4, 8, 12]
radii = [1, 10, 50]

block_matching_cases = itertools.product(block_sizes, [0, 10], [1, 10], radii)


def to_image(A):
    # If all are equal
    B = np.array(A).astype(float)
    if ~np.any(B - B.max()):
        return B.astype(np.uint16)
    return (B / B.max()).astype(np.uint16) * (2 ** 16 - 1)


def add_noise(A, noise_level):

    return to_image(A) + to_image(noise_level * np.random.rand(*np.shape(A)))


def create_circle_data(
    x_start=200,
    x_end=250,
    y_start=250,
    y_end=250,
    r=30,
    line_x=None,
    line_y=None,
    Nx=500,
    Ny=500,
):

    if line_x is None:
        line_x = np.sin(np.linspace(0, np.pi, 100))
    if line_x is None:
        line_y = np.sin(np.linspace(0, np.pi, 100))

    X = x_start + np.multiply((x_end - x_start), line_x)
    Y = y_start + np.multiply((y_end - y_start), line_y)

    A = []
    for x, y in zip(X, Y):
        a = np.fromfunction(
            lambda i, j: np.sqrt((i - x) ** 2 + (j - y) ** 2) < r, (Nx, Ny)
        )
        A.append(a)

    return to_image(A).T


def test_create_data():

    plot = False
    x_start = 200
    x_end = 250
    y_start = 250
    y_end = 250
    r = 30
    line_x = [0, 1]
    line_y = [0, 1]
    Nx = 500
    Ny = 500
    A = create_circle_data(
        x_start=x_start,
        x_end=x_end,
        y_start=y_start,
        y_end=y_end,
        r=r,
        line_x=line_x,
        line_y=line_y,
        Nx=Nx,
        Ny=Ny,
    )
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(A[:, :, 0], cmap=cm.gray)
    ax[0].set_title(f"First frame no noise")
    ax[1].imshow(A[:, :, 1], cmap=cm.gray)
    ax[1].set_title(f"Second frame no noise")
    for axi in ax:
        axi.grid(True)

    noise_level = 0.004
    B = add_noise(A, noise_level)

    if plot:
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(B[:, :, 0], norm=LogNorm(), cmap=cm.gray)
        ax[0].set_title(f"First frame noise level {noise_level} (lognorm)")
        ax[1].imshow(B[:, :, 1], cmap=cm.gray)
        ax[1].set_title(f"Second frame noise level {noise_level}")
        for axi in ax:
            axi.grid(True)
        plt.show()


@pytest.mark.parametrize("block_size, dx, dy, r", block_matching_cases)
def test_block_matching_single_circle(block_size, dx, dy, r):

    x_start = 200
    x_end = x_start + dx
    y_start = 250
    y_end = y_start + dy

    Nx = 500
    Ny = 500
    N = 3

    mps_data = sample_data(
        x_start=x_start,
        x_end=x_end,
        y_start=y_start,
        y_end=y_end,
        r=r,
        N=N,
        Nx=Nx,
        Ny=Ny,
    )
    A = mps_data.frames

    p = max(abs(dx), abs(dy))

    # If both images are the same then all should be zero
    vectors = mc.motion_tracking.block_matching(
        A[:, :, 0], A[:, :, 0], block_size=block_size, max_block_movement=p
    )
    assert np.all(vectors == 0.0)
    vectors = mc.motion_tracking.block_matching(
        A[:, :, 0], A[:, :, 1], block_size=block_size, max_block_movement=p
    )
    amp = np.linalg.norm(vectors, axis=2)
    print(f"Max X: {vectors[:, :, 0].max()}, Min X: {vectors[:, :, 0].min()}")
    print(f"Max Y: {vectors[:, :, 1].max()}, Min Y: {vectors[:, :, 1].min()}")
    print(f"Max amp: {amp.max()}, Min amp: {amp.min()}")

    if y_end >= y_start:
        assert vectors[:, :, 0].max() == dy
    else:
        assert vectors[:, :, 0].min() == dy
    if x_end >= x_start:
        assert vectors[:, :, 1].max() == dx
    else:
        assert vectors[:, :, 1].min() == dx

    assert amp.max() == np.linalg.norm([dx, dy])


def plot_block_matching_double_circle():

    block_size = 8
    dx = 50
    dy = 0
    r = 30

    x_start = 200
    y_start = 250

    line_x = [0, 1]
    line_y = [0, 1]
    Nx = 500
    Ny = 500
    A = create_circle_data(
        x_start=x_start,
        x_end=x_start + dx,
        y_start=y_start,
        y_end=y_start + dy,
        r=r,
        line_x=line_x,
        line_y=line_y,
        Nx=Nx,
        Ny=Ny,
    )

    A += create_circle_data(
        x_start=x_start + 5 * r,
        x_end=x_start - dx + 5 * r,
        y_start=y_start + 5 * r,
        y_end=y_start - dy + 5 * r,
        r=r,
        line_x=line_x,
        line_y=line_y,
        Nx=Nx,
        Ny=Ny,
    )

    p = max(abs(dx), abs(dy))

    vectors = mc.motion_tracking.block_matching(
        A[:, :, 0], A[:, :, 1], block_size=block_size, max_block_movement=p
    )
    amp = np.linalg.norm(vectors, axis=2)

    print(f"Max X: {vectors[:, :, 0].max()}, Min X: {vectors[:, :, 0].min()}")
    print(f"Max Y: {vectors[:, :, 1].max()}, Min Y: {vectors[:, :, 1].min()}")
    print(f"Max amp: {amp.max()}, Min amp: {amp.min()}")
    # if y_end >= y_start:
    #     assert vectors[:, :, 0].min() == -dy
    # else:
    #     assert vectors[:, :, 0].max() == -dy
    # if x_end >= x_start:
    #     assert vectors[:, :, 1].min() == -dx
    # else:
    #     assert vectors[:, :, 1].max() == -dx

    # assert amp.max() == np.linalg.norm([dx, dy])

    if 1:
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(A[:, :, 0])
        ax[0].set_title(f"First frame no noise")
        ax[1].imshow(A[:, :, 1])
        ax[1].set_title(f"Second frame no noise")
        for axi in ax:
            axi.grid(True)

        fig, ax = plt.subplots(2, 2)

        ax[0, 0].imshow(A[:, :, 0])
        im = ax[0, 0].imshow(A[:, :, 1], alpha=0.5)
        ax[0, 0].grid(True)
        ax[0, 0].set_title("Data")
        fig.colorbar(im, ax=ax[0, 0])

        im = ax[0, 1].imshow(amp)
        ax[0, 1].grid(True)
        ax[0, 1].set_title("Amplitude")

        fig.colorbar(im, ax=ax[0, 1])

        im = ax[1, 0].imshow(vectors[:, :, 0])
        ax[1, 0].grid(True)
        ax[1, 0].set_title("X vector")
        fig.colorbar(im, ax=ax[1, 0])

        im = ax[1, 1].imshow(vectors[:, :, 1])
        ax[1, 1].grid(True)
        ax[1, 1].set_title("Y vector")
        fig.colorbar(im, ax=ax[1, 1])
        fig.tight_layout()

        fig, ax = plt.subplots()
        x = np.linspace(0, Nx, vectors.shape[0])
        y = np.linspace(0, Ny, vectors.shape[1])
        # X, Y = np.meshgrid(x, y)
        ax.quiver(
            y,
            x,
            -vectors[:, :, 1],
            vectors[:, :, 0],
            angles="xy",
            scale_units="xy",
            scale=1,
        )
        ax.set_aspect("equal")
        plt.show()


@pytest.mark.parametrize("block_size, dx, dy, r", block_matching_cases)
def test_block_matching_single_circle(block_size, dx, dy, r):

    x_start = 200
    x_end = x_start + dx
    y_start = 250
    y_end = y_start + dy

    Nx = 500
    Ny = 500
    N = 3

    mps_data = sample_data(
        x_start=x_start,
        x_end=x_end,
        y_start=y_start,
        y_end=y_end,
        r=r,
        N=N,
        Nx=Nx,
        Ny=Ny,
    )
    A = mps_data.frames

    p = max(abs(dx), abs(dy))

    # If both images are the same then all should be zero
    vectors = mc.motion_tracking.block_matching(
        A[:, :, 0], A[:, :, 0], block_size=block_size, max_block_movement=p
    )
    assert np.all(vectors == 0.0)
    vectors = mc.motion_tracking.block_matching(
        A[:, :, 0], A[:, :, 1], block_size=block_size, max_block_movement=p
    )
    amp = np.linalg.norm(vectors, axis=2)
    print(f"Max X: {vectors[:, :, 0].max()}, Min X: {vectors[:, :, 0].min()}")
    print(f"Max Y: {vectors[:, :, 1].max()}, Min Y: {vectors[:, :, 1].min()}")
    print(f"Max amp: {amp.max()}, Min amp: {amp.min()}")

    if y_end >= y_start:
        assert vectors[:, :, 0].max() == dy
    else:
        assert vectors[:, :, 0].min() == dy
    if x_end >= x_start:
        assert vectors[:, :, 1].max() == dx
    else:
        assert vectors[:, :, 1].min() == dx

    assert amp.max() == np.linalg.norm([dx, dy])


def plot_block_matching():

    plot = True
    block_size = 8
    dx = 10
    dy = 10
    r = 50

    x_start = 200
    x_end = x_start + dx
    y_start = 250
    y_end = y_start + dy

    line_x = [0, 1]
    line_y = [0, 1]
    Nx = 500
    Ny = 500
    A = create_circle_data(
        x_start=x_start,
        x_end=x_end,
        y_start=y_start,
        y_end=y_end,
        r=r,
        line_x=line_x,
        line_y=line_y,
        Nx=Nx,
        Ny=Ny,
    )
    A = add_noise(A, 0.004)
    p = 2 * max(abs(dx), abs(dy))

    vectors = mc.motion_tracking.block_matching(
        A[:, :, 0], A[:, :, 1], block_size=block_size, max_block_movement=p
    )
    amp = np.linalg.norm(vectors, axis=2)

    print(f"dx: {dx}, dy: {dy}, Max move: {p}")
    print(f"Max X: {vectors[:, :, 0].max()}, Min X: {vectors[:, :, 0].min()}")
    print(f"Max Y: {vectors[:, :, 1].max()}, Min Y: {vectors[:, :, 1].min()}")
    print(f"Max amp: {amp.max()}, Min amp: {amp.min()}")

    if plot:
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(A[:, :, 0])
        ax[0].set_title(f"First frame no noise")
        ax[1].imshow(A[:, :, 1])
        ax[1].set_title(f"Second frame no noise")
        for axi in ax:
            axi.grid(True)

        fig, ax = plt.subplots(2, 2)

        ax[0, 0].imshow(A[:, :, 0])
        im = ax[0, 0].imshow(A[:, :, 1], alpha=0.5)
        ax[0, 0].grid(True)
        ax[0, 0].set_title("Data")
        fig.colorbar(im, ax=ax[0, 0])

        im = ax[0, 1].imshow(amp)
        ax[0, 1].grid(True)
        ax[0, 1].set_title("Amplitude")

        fig.colorbar(im, ax=ax[0, 1])

        im = ax[1, 0].imshow(vectors[:, :, 0])
        ax[1, 0].grid(True)
        ax[1, 0].set_title("X vector")
        fig.colorbar(im, ax=ax[1, 0])

        im = ax[1, 1].imshow(vectors[:, :, 1])
        ax[1, 1].grid(True)
        ax[1, 1].set_title("Y vector")
        fig.colorbar(im, ax=ax[1, 1])
        fig.tight_layout()

        fig, ax = plt.subplots()
        x = np.linspace(0, Nx, vectors.shape[0])
        y = np.linspace(0, Ny, vectors.shape[1])
        # X, Y = np.meshgrid(x, y)
        vy = vectors[:, :, 1]
        vx = vectors[:, :, 0]

        # import scipy.interpolate as interp
        # import scipy.integrate as integrate
        # dfunx = interp.interp2d(X[:], X[:], vx[:])
        # dfuny = interp.interp2d(Y[:], Y[:], vy[:])
        # dfun = lambda xy,t: [dfuny(xy[0], xy[1])[0],
        #                      dfunx(xy[0], xy[1])[0]]

        # p0 = np.array([0.5, 0.5])
        # dt = 0.01
        # t0 = 0
        # t1 = 1
        # t = np.arange(t0, t1 + dt, dt)

        # streamline = integrate.odeint(dfun, p0, t)
        # from scipy.interpolate import griddata

        # grid_z0 = griddata(points, values, (grid_x, grid_y), method='nearest')
        # grid_z1 = griddata(points, values, (grid_x, grid_y), method='linear')
        # grid_z2 = griddata(points, values, (grid_x, grid_y), method='cubic')
        # points =  np.array(list(zip(x.flatten(),y.flatten())))
        # grid_x = griddata(points, vx.flatten(), (grid_x, grid_y), method='nearest')

        # print(streamline)

        # ax.plot(streamline[:, 0], streamline[:, 1])
        ax.quiver(y, x, vy, vx, angles="xy", scale_units="xy", scale=1)
        ax.set_aspect("equal")
        plt.show()
        plt.clf()


def plot_displacements():

    dx = 50
    dy = 50
    r = 30
    block_size = 8
    x_start = 200
    x_end = x_start + dx
    y_start = 250
    y_end = y_start + dy

    Nx = 500
    Ny = 500
    N = 100
    mps_data = sample_data(
        x_start=x_start,
        x_end=x_end,
        y_start=y_start,
        y_end=y_end,
        r=r,
        N=N,
        Nx=Nx,
        Ny=Ny,
    )
    A = mps_data.frames
    max_block_movement = max(dx, dy)
    motion = mc.MotionTracking(
        mps_data,
        block_size=block_size,
        max_block_movement=max_block_movement,
    )

    all_vectors = motion.displacement_vectors

    idx = 0
    vectors = all_vectors[:, :, :, idx]

    img = A[:, :, idx]

    fig, ax = plt.subplots()
    x = np.linspace(0, Nx, vectors.shape[0])
    y = np.linspace(0, Ny, vectors.shape[1])

    im = ax.imshow(img, cmap=cm.gray_r)
    Q = ax.quiver(
        y,
        x,
        all_vectors[:, :, 1, idx],
        -all_vectors[:, :, 0, idx],
        color="r",
        units="xy",
        scale_units="inches",
        scale=10,
    )
    ax.set_aspect("equal")

    def update(idx):
        Q.set_UVC(all_vectors[:, :, 1, idx], -all_vectors[:, :, 0, idx])
        im.set_array(A[:, :, idx])

    # Set up formatting for the movie files
    Writer = animation.writers["ffmpeg"]
    writer = Writer(fps=mps_data.framerate, metadata=dict(artist="Me"), bitrate=1800)

    anim = animation.FuncAnimation(fig, update, N - 1, interval=50, blit=False)

    anim.save("displacement.mp4", writer=writer)


def sample_data(
    x_start=200, x_end=250, y_start=250, y_end=300, r=30, N=20, Nx=500, Ny=500
):
    x = np.linspace(0, np.pi, N)
    line_x = np.sin(x)
    line_y = np.sin(x)

    A = create_circle_data(
        x_start=x_start,
        x_end=x_end,
        y_start=y_start,
        y_end=y_end,
        r=r,
        line_x=line_x,
        line_y=line_y,
        Nx=Nx,
        Ny=Ny,
    )

    framerate = N
    dt = 1000.0 / framerate
    info = dict(um_per_pixel=1.0, time_unit="ms")
    time_stamps = np.arange(0, N * dt, dt)
    return MPSData(
        frames=A, time_stamps=time_stamps, info=info, framerate=framerate, num_frames=N
    )


def test_displacements():

    dx = 30
    dy = 30
    r = 10

    x_start = 100
    y_start = 100

    line_x = line_y = np.array([0, 0.5, 1.0, 0.5, 0])
    N = len(line_x)
    Nx = 500
    Ny = 500

    A = np.zeros((Nx, Ny, N))

    for rx in range(1, 30, 7):
        for ry in range(1, 30, 7):
            x0 = x_start + rx * r
            y0 = y_start + ry * r
            x1 = x0 - dx
            y1 = y0 - dy

            A += create_circle_data(
                x_start=x0,
                x_end=x1,
                y_start=y0,
                y_end=y1,
                r=r,
                line_x=line_x,
                line_y=line_y,
                Nx=Nx,
                Ny=Ny,
            )

    framerate = N
    dt = 1000.0 / framerate
    info = dict(um_per_pixel=1.0, time_unit="ms")
    time_stamps = np.arange(0, N * dt, dt)
    mps_data = MPSData(
        frames=A, time_stamps=time_stamps, info=info, framerate=framerate, num_frames=N
    )

    motion = mc.MotionTracking(
        mps_data,
        max_block_movement=max(abs(dx), abs(dy)),
        block_size=8,
        filter_kernel_size=0,
    )

    x_displacement = dx * line_x
    y_displacement = dy * line_y

    data = motion.displacement_vectors
    xmax = np.max(data[:, 0], axis=(0, 1))
    ymax = np.max(data[:, 1], axis=(0, 1))

    factor_x = np.divide(xmax, x_displacement, where = x_displacement != 0)
    factor_y = np.divide(ymax, y_displacement, where = y_displacement != 0)

    assert np.nanmean(factor_x) < 1e-12
    assert np.nanmean(factor_y) < 1e-12


if __name__ == "__main__":
    test_block_matching_single_circle(8, 10, 10, 10)
    #plot_block_matching()
    #plot_block_matching_double_circle()
    test_create_data()
    #plot_displacements()
    test_displacements()
