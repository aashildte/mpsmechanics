import os
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mps

from mpsmechanics import (
    read_prev_layer,
    analyze_mechanics,
    calc_norm_over_time,
)


def plot_over_area(
    strain_data, im_data, time, M, N, label, title, output_pref
):

    xshape, yshape = strain_data.shape[1:3]

    blocksize = im_data.shape[0] // xshape

    xlen, ylen = xshape // M, yshape // N
    area = xlen * ylen

    fig, axes = plt.subplots(
        M, N, sharex=True, sharey=True, figsize=(6 * M, 12 * N)
    )

    data_over_area = []

    for m in range(1, M + 1):
        data_over_ns = []
        for n in range(1, N + 1):
            xfrom = (m - 1) * xlen
            yfrom = (n - 1) * ylen
            data_over_ns.append(
                1
                / area
                * calc_norm_over_time(
                    strain_data[
                        :, xfrom : xfrom + xlen, yfrom : yfrom + ylen
                    ]
                )
            )
        data_over_area.append(data_over_ns)

    for m in range(M):
        for n in range(N):
            if M == 1:
                axes[n].plot(time, data_over_area[m][n])
            elif N == 1:
                axes[m].plot(time, data_over_area[m][n])
            else:
                axes[m, n].plot(time, data_over_area[m][n])

    frame = fig.add_subplot(111, frameon=False)
    frame.tick_params(
        labelcolor="none",
        top=False,
        bottom=False,
        left=False,
        right=False,
    )

    frame.set_title(title)
    frame.set_ylabel(label)
    frame.set_xlabel("Time (ms)")

    plt.savefig(f"{output_pref}_overtime_{M}_{N}.png")
    plt.clf()

    plt.imshow(im_data, cmap="gray")

    for m in range(0, M + 1):
        plt.plot(
            [0, im_data.shape[1] - 1],
            [m * xlen * blocksize, m * xlen * blocksize],
            "w",
            linewidth=4,
        )

    for n in range(0, N + 1):
        plt.plot(
            [n * ylen * blocksize, n * ylen * blocksize],
            [0, im_data.shape[0] - 1],
            "w",
            linewidth=4,
        )

    plt.savefig(f"{output_pref}_image_{M}_{N}.png")


f_in = sys.argv[1]
M = int(sys.argv[2])
N = int(sys.argv[3])
output_folder = sys.argv[4]

data = read_prev_layer(f_in, analyze_mechanics)

im_data = mps.MPS(f_in).frames[:, :, 0]

strain_data = data["all_values"]["principal_strain"]
time = data["time"]

plot_over_area(
    strain_data,
    im_data,
    time,
    M,
    N,
    "Strain (-)",
    "Principal strain",
    os.path.join(output_folder, "strain"),
)
