"""Lines of equal redshift in the observer plane"""

import numpy as np
from luminet.spatial import polar_cartesian_distance
import logging
logger = logging.getLogger(__name__)


class Isoredshift:
    """Leightwieght dataclass to store and visualize lines of equal redshift in the observer plane.

    This class is rarely initialized by the user. It is however often used by the :class:`~luminet.black_hole.BlackHole`
    class to handle isoredshifts.

    Isoredshift lines usually have two solutions, more or less in each hemisphere of the observer plane.
    A notable exception is closed isoredshifts, which will have only a single solution at the tip.
    """
    def __init__(
        self,
        redshift,
        order = 0,
        angles = None,
        impact_parameters = None,
        ir_radii = None
    ):
        """
        Args:
            incl (float): Inclination angle of the observer
            redshift (float): redshift value
            bh_mass (float): mass of the black hole
        """
        self.redshift = redshift
        """float: redshift value"""
        self.order = order
        """int: order of the image associated with this isoredshift"""

        # Isoredshift attributes: normally set by BlackHole
        self.angles = None
        """np.ndarray: angles of the isoredshifts"""
        self.impact_parameters = None
        """np.ndarray: radii of the isoredshifts in the observer plane."""
        self.ir_radii = ir_radii or None
        """np.ndarray: radii of the isoradials used to calculate the isoredshifts"""

        if angles is not None: self.set_angles(angles)
        if impact_parameters is not None: self.set_impact_parameters(impact_parameters)

    def set_angles(self, angle_pairs):
        r"""Set the angular coordinates of this isoredshift
        
        Convenience method to initialize coordinates from the solver results.
        Solver results return two-tuples, which are unpacked here.

        Args:
            angle_pairs (List[tuple]): Array of two-tuples of angles :math:`\alpha`.
        """
        self.angles = np.array(list(zip(*angle_pairs)))

    def set_impact_parameters(self, impact_parameter_pairs):
        """Set the observed radial coordinates of this isoredshift
        
        Convenience method to initialize coordinates from the solver results.
        Solver results return two-tuples, which are unpacked here.

        Args:
            impact_parameter_pairs (List[tuple]): Array of two-tuples of impact parameters :math:`b`.
        """
        self.impact_parameters = np.array(list(zip(*impact_parameter_pairs)))

    def _clean(self):
        """Remove None values from the coordinates"""
        nanmask0 = [a != None for a in self.angles[0]]
        angles0 = self.angles[0][nanmask0]
        impact_parameters0 = self.impact_parameters[0][nanmask0]

        nanmask1 = [a != None for a in self.angles[1]]
        angles1 = self.angles[1][nanmask1]
        impact_parameters1 = self.impact_parameters[1][nanmask1]

        self.angles = np.array([angles0, angles1])
        self.impact_parameters = np.array([impact_parameters0, impact_parameters1])


    def _get_last_points(self):
        """Get the last not-None coordinate of each subredshift
        """
        a1 = [a for a in self.angles[0] if a is not None]
        a1 = a1[-1] if a1 else None
        b1 = [b for b in self.impact_parameters[0] if b is not None]
        b1 = b1[-1] if b1 else None
        p1 = (a1, b1)

        a2 = [a for a in self.angles[1] if a is not None]
        a2 = a2[-1] if a2 else None
        b2 = [b for b in self.impact_parameters[1] if b is not None]
        b2 = b2[-1] if b2 else None
        p2 = (a2, b2)

        return p1, p2

    def _is_close(self, tol=5e-2):
        """Check if the isoredshift is (almost) a closed one.

        Args:
            tol (float): 
                Tolerance below which the gap between the two 
                sub-isoredshifts is considered to be closed.

        Returns:
            bool: True if the Euclidean distance between the two end-points is below :paramref:`tol`
        """
        # last not-None coordinates of each subredshift
        p1, p2 = self._get_last_points()
        if any([c is None for c in [*p1, *p2]]): return False
        if polar_cartesian_distance(p1, p2) < tol:
            return True
        return False
    
    def _join(self):
        """Join the isoredshift if it is (almost) a closed one."""
        self._clean()
        self.angles = np.array([*self.angles[0], *self.angles[1][::-1]]), np.array([])
        self.impact_parameters = np.array([*self.impact_parameters[0], *self.impact_parameters[1][::-1]]), np.array([])
    
    def plot(self, ax, **kwargs):
        """Plot the isoredshift on an ax
        
        Args:
            ax (:class:`matplotlib.axes.Axes`): Ax object to plot on.
            kwargs (optional): Optional keyword arguments to pass to :func:`matplotlib.pyplot.plot`
        """
        if self._is_close(): self._join()
        for n in range(len(self.angles)):
            ax.plot(self.angles[n], self.impact_parameters[n], label=self.redshift, **kwargs)
        return ax