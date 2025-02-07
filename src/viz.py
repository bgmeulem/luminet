"""Visualization routines for the project.

Provides convenience functions for plotting colored lines.
"""

import matplotlib.collections as mcoll
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def colorline(ax, x, y, z, norm, cmap, linewidth=3, **kwargs):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """
    cmap = cm.get_cmap(cmap)
    norm = plt.Normalize(*norm)
    segments = make_segments(x, y)
    lc = mcoll.LineCollection(
        segments,
        cmap=cmap,
        linewidth=linewidth,
        capstyle="round",
        norm=norm,
        **kwargs
    )
    lc.set_array(z)
    ax.add_collection(lc)
    # mx = max(segments[:][:, 1].flatten())
    # _ax.set_ylim((0, mx))
    return ax
