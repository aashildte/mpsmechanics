"""

Åshild Telle / Simula Research Laboratory / 2019

"""

import matplotlib.pyplot as plt


def make_pretty_label(key, unit):
    """

    Consistent labels with capital letters, no underscores and units in ().

    Args:
        key - which quantity
        unit - corresponding unit

    """

    label = key.capitalize()
    return label.replace("_", " ") + f" ({unit})"


def setup_frame(num_rows, num_cols, sharex, sharey):
    """

    Setup of subplots, gives similar configuration across different
    kinds of plots. Aligns axes.

    Args:
        num_rows - number of rows among the subplots
        num_cols - number of columns

    """

    figsize = (14, 12)
    dpi = 300
    fig, axes = plt.subplots(num_rows, num_cols, \
                             sharex=sharex, sharey=sharey, \
                             figsize=figsize, dpi=dpi, squeeze=False)

    axes = axes.flatten()
    fig.align_ylabels(axes)

    return axes, fig


def get_plot_fun(values, corr_plots):
    """

    Visualization/plotting scripts are usually performed over 1D, 2D
    or 4D values (shape (), (2,) or (2, 2)). Given one of these
    shapes, and a list of corresponding functions which performs the
    plotting, this functions performs the mapping between these two.

    Args:
        values - to be plotted, numpy array of dim. T x X x Y X D
        corr_plots - list of 3 different plotting functions

    """

    num_dims = values.shape[3:]

    assert num_dims in ((), (2,), (2, 2)), \
        f"Error: shape of {num_dims} not recognized."

    plot_1d_values, plot_2d_values, plot_4d_values = corr_plots

    if num_dims == ():
        return plot_1d_values

    if num_dims == (2,):
        return plot_2d_values

    return plot_4d_values
