"""Visualization routines for the project.

Provides convenience functions for plotting colored lines.
"""

import matplotlib.collections as mcoll
import matplotlib.pyplot as plt
import numpy as np

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
    """Plot a line that changes color along the way.

    Args:
        ax (:class:`~matplotlib.axes.Axes`): Where to plot the line on
        x (array-like): x-coordinates (or angles in polar)
        y (array-like): y-coordinates (or radii in polar)
        z (array-like): color values.
        norm (tuple): Min and max of the colorscale.
        cmap (str): Name of the colormap.
        linewidth (float): width of the line.
        kwargs (optional): Additional keyword arguments to pass to :class:`matplotlib.collections.LineCollection`.

    See also:
        http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    
    See also:
        http://matplotlib.org/examples/pylab_examples/multicolored_line.html

    Returns:
        :class:`~matpltolib.axes.Axes`: Axes with plotted line.

    """
    cmap = plt.get_cmap(cmap)
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
