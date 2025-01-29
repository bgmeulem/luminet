from .isoradial import Isoradial
from . import transform
from matplotlib import tri
from matplotlib.axes import Axes
from typing import Callable

def _get_delauney_tri(
    a, b, z,
    mask: Callable | None = None, 
    order=0
):
    assert order >= 0
    x, y = transform.polar_to_cartesian(a, b)
    delauney = tri.Triangulation(x, y)
    ref_field = z
    delauney, ref_field = tri.UniformTriRefiner(delauney).refine_field(ref_field, subdiv=3)
    if mask is not None:
        delauney.set_mask(mask(delauney))
    return delauney, ref_field

def refine_polar_triangulation(triangulation, ref_field, subdiv):
    triangulation.x, triangulation.y = transform.polar_to_cartesian(triangulation.x, triangulation.y)
    triangulation, ref_field = tri.UniformTriRefiner(triangulation).refine_field(ref_field, subdiv=subdiv)
    triangulation.x, triangulation.y = transform.cartesian_to_polar(triangulation.x, triangulation.y)
    return triangulation, ref_field


def plot_points(
    points, 
    ax, 
    mask: Callable | None = None, 
    levels=100
) -> Axes:
    """
    Plot the points written out by samplePoints()
    """

    max_flux = max(points['flux_o'])
    min_flux = 0

    points.sort_values(by="flux_o", inplace=True, ascending=True)
    
    z = points["flux_o"]
    a, b = points["alpha"], points["impact_parameter"]
    delauney, ref_flux_o = _get_delauney_tri(
        a, b,
        z,
        mask=mask)
    a, b = transform.cartesian_to_polar(delauney.x, delauney.y)
    ax.tricontourf(
        a, b, ref_flux_o, 
        triangles=delauney.get_masked_triangles(), 
        zorder=1, 
        cmap="Greys_r",
        vmin = min_flux,
        vmax = max_flux,
        levels=levels
        )
    return ax
