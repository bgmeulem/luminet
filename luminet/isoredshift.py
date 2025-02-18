"""Lines of equal redshift in the observer plane"""

import numpy as np
from typing import List
from luminet.isoradial import Isoradial
import logging
logger = logging.getLogger(__name__)


class Isoredshift:
    """Class to calculate and visualize lines of equal redshift in the observer plane.
    """
    def __init__(
        self,
        incl,
        redshift,
        bh_mass,
        from_isoradials=None,
    ):
        """
        Args:
            inclination (float): inclination angle of the observer
            redshift (float): redshift value
            bh_mass (float): mass of the black hole
            from_isoradials (List[Isoradial]): 
                :py:class:`Isoradial` objects used to calculate the isoredshifts.
                Note that the coordinates of these isoradials do not need to be computed.
        """
        # Parent black hole parameters
        if from_isoradials is None:
            from_isoradials = {}

        self.incl = incl
        """float: inclination angle of the observer"""
        self.bh_mass = bh_mass
        """float: mass of the black hole"""
        self.redshift = redshift
        """float: redshift value"""

        # Isoredshift attributes
        self.angles = np.array([np.empty_like(from_isoradials, dtype=float), np.empty_like(from_isoradials, dtype=float)])
        """np.ndarray: angles of the isoredshifts"""
        self.radii_b = np.array([np.empty_like(from_isoradials, dtype=float), np.empty_like(from_isoradials, dtype=float)])
        """np.ndarray: radii of the isoredshifts in the observer plane."""
        self.ir_radii = np.empty_like(from_isoradials, dtype=float)
        """np.ndarray: radii of the isoradials used to calculate the isoredshifts"""

        # Calculate coordinates
        self.calc_from_isoradials(from_isoradials)

    def calc_from_isoradials(self, isoradials: List[Isoradial]):
        """Calculate the isoredshift for a single redshift value, based on their intersection with isoradials.

        Args:
            isoradials (List[Isoradial]): isoradials used to calculate the isoredshifts

        Returns:
            :py:class:`Isoredshift`: The :py:class:`Isoredshift` object itself, but with calculated coordinates.
        """
        for i, ir in enumerate(isoradials):
            a, b = ir.calc_redshift_locations(self.redshift)
            if all([e is None for e in a]):
                logger.warning("Isoredshift for z={} is initialized from isoradial R={} that does not contain this reedshift.".format(self.redshift, ir.radius))
                break
            self.ir_radii[i] = ir.radius
            for solution_index in range(len(a)):
                self.angles[solution_index][i] = a[solution_index]
                self.radii_b[solution_index][i] = b[solution_index]
        return self

    def calc_from_optimize(self):
        """
        Calculates the isoredshift for a single redshift value, based on a couple of isoradials calculated
        at low precision

        :meta private:
        """
        init_ir_radius = 6*self.bh_mass
        ir = Isoradial(init_ir_radius, self.incl, self.bh_mass)
        ir.calc_redshift_locations(self.redshift)

        

    def plot(self, ax, **kwargs):
        for n in range(len(self.angles)):
            ax.plot(self.angles[n], self.radii_b[n], label=self.redshift, **kwargs)
        return ax