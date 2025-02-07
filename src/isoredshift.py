import numpy as np
from typing import List
from .isoradial import Isoradial
import logging
logger = logging.getLogger(__name__)


class Isoredshift:
    def __init__(
        self,
        inclination,
        redshift,
        bh_mass,
        solver_parameters=None,
        from_isoradials=None,
    ):
        # Parent black hole parameters
        if from_isoradials is None:
            from_isoradials = {}

        self.t = inclination
        self.M = bh_mass
        self.t = inclination
        self.redshift = redshift

        # Isoredshift attributes
        self.angles = np.array([np.empty_like(from_isoradials, dtype=float), np.empty_like(from_isoradials, dtype=float)])
        self.radii_b = np.array([np.empty_like(from_isoradials, dtype=float), np.empty_like(from_isoradials, dtype=float)])
        self.ir_radii = np.empty_like(from_isoradials, dtype=float)

        # Calculate coordinates
        self.calc_from_isoradials(from_isoradials)

    def calc_from_isoradials(self, isoradials: List[Isoradial]):
        """
        Calculates the isoredshift for a single redshift value, based on a couple of isoradials calculated
        at low precision
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
        """
        init_ir_radius = 6*self.M
        ir = Isoradial(init_ir_radius, self.t, self.M)
        ir.calc_redshift_locations(self.redshift)

        

    def plot(self, ax, **kwargs):
        for n in range(len(self.angles)):
            ax.plot(self.angles[n], self.radii_b[n], label=self.redshift, **kwargs)
        return ax