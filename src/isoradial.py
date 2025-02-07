import configparser
from typing import Dict, List
import matplotlib.pyplot as plt

import numpy as np

from . import black_hole_math as bhmath
from . import transform
from .solver import improve_solutions
from .viz import colorline


class Isoradial:
    def __init__(
        self,
        radius: float,
        incl: float,
        bh_mass: float,
        order: int = 0,
        acc: float | None = None,
        params: Dict | None = None,
    ):
        assert (
            radius >= 6.0 * bh_mass
        ), """
        Radius should be at least 6 times the mass of the black hole. 
        No orbits are stable below this radius for Swarzschild black holes.
        """
        self.bh_mass = bh_mass  # mass of the black hole containing this isoradial
        self.incl = incl  # inclination of observer's plane
        self.radius = radius
        self.order = order
        self.acc = acc
        self.params = params if params is not None else {"angular_precision": 100}
        self.find_redshift_params = {
            "force_redshift_solution": False,  # force finding a redshift solution on the isoradial
            "max_force_iter": 5,  # only make this amount of iterations when forcing finding a solution
        }

        self.radii_b = []
        self.angles = []
        self.redshift_factors = None

        self.calculate()

    def calculate_coordinates(self):
        """Calculates the angles (alpha) and radii (b) of the photons emitted at radius self.radius as they would appear
        on the observer's photographic plate. Also saves the corresponding values for the impact parameters (P).

        Args:

        Returns:
            tuple: Tuple containing the angles (alpha) and radii (b) for the image on the observer's photographic plate
        """

        angles = []
        impact_parameters = []
        t = np.linspace(0, 2 * np.pi, self.params["angular_precision"])
        for alpha in t:
            b = bhmath.calc_impact_parameter(
                radius=self.radius,
                incl=self.incl,
                alpha=alpha,
                bh_mass=self.bh_mass,
                order=self.order,
            )
            assert (
                b > 0
            ), "Impact parameter should be positive, but it wasnt for: R={}, alpha={}, incl={}".format(
                self.radius, alpha, self.incl
            )
            if b is np.nan:
                impact_parameters.append(bhmath.ellipse(self.radius, alpha, self.incl))
            else:
                impact_parameters.append(b)
            angles.append(alpha)

        # flip image if necessary
        self.angles = np.array(angles)
        self.radii_b = np.array(impact_parameters)
        return angles, impact_parameters

    def calc_redshift_factors(self):
        """Calculates the redshift factor (1 + z) over the line of the isoradial"""
        redshift_factors = bhmath.redshift_factor(
            radius=self.radius,
            angle=self.angles,
            incl=self.incl,
            bh_mass=self.bh_mass,
            b=self.radii_b,
        )
        self.redshift_factors = np.array(redshift_factors)
        return redshift_factors

    def calculate(self):
        self.calculate_coordinates()
        self.calc_redshift_factors()
        return self

    def get_b_from_angle(self, angle: float):
        angles_array = np.array(self.angles)
        radii_b_array = np.array(self.radii_b)

        def find_closest_radii(ang):
            indices = np.argmin(np.abs(angles_array - ang), axis=-1)
            return radii_b_array[indices]

        if isinstance(angle, (float, int)):
            return find_closest_radii(angle)
        else:
            angle = np.asarray(angle)
            return np.vectorize(find_closest_radii, otypes=[float])(angle)

    def calc_b_from_angle(self, angle: float):
        b = bhmath.calc_impact_parameter(
            radius=self.radius,
            incl=self.incl,
            alpha=angle,
            bh_mass=self.bh_mass,
            order=self.order,
        )
        return b

    def plot(self, ax, z, cmap, norm, **kwargs):
        """

        Args:
            ax: The axis on which the isoradial should be plotted
            z: The color values to be used
            cmap (str): The colormap to be used
            norm (tuple): The normalization to be used
            **kwargs: Additional arguments to be passed to the colorline function
        """
        ax = colorline(
            ax, self.angles, self.radii_b, z=z, cmap=cmap, norm=norm, **kwargs
        )

        return ax


    def _has_redshift(self, z):
        """Calculate if the class theoretically contains redshift value :math:`z`"""
        # TODO: math

    def calc_redshift_locations(self, redshift):
        """
        Calculates which location on the isoradial has some redshift value (not redshift factor)

        """

        func = (
            lambda angle: redshift
            + 1
            - bhmath.redshift_factor(
                self.radius,
                angle,
                self.incl,
                self.bh_mass,
                self.calc_b_from_angle(angle),
            )
        )

        # Find all zero crossings
        angles = np.linspace(0, 2 * np.pi, 1000)
        values = [func(angle) for angle in angles]
        sign_changes = np.where(np.diff(np.sign(values)))[0]
        
        solutions = [None, None]  # Initialize with None to ensure two solutions
        for idx in sign_changes:
            # Find the root of the function in the interval
            angle0, angle1 = angles[idx], angles[idx + 1]
            y0, y1 = values[idx], values[idx + 1]
            try:
                root = improve_solutions(func, (angle0, angle1), (y0, y1), kwargs={})
                if y0 < 0 and y1 > 0:
                    solutions[0] = root  # Negative to positive
                elif y0 > 0 and y1 < 0:
                    solutions[1] = root  # Positive to negative
            except Exception as e:
                # If brentq fails to find a root in the interval, skip it
                pass
        
        # Calculate corresponding b values for the found angles
        b_values = [self.calc_b_from_angle(angle) if angle is not None else None for angle in solutions]
        
        assert len(b_values) == len(solutions) == 2, "Should have found 2 solutions, or at least padded the second solution with None"
        return np.array(solutions), np.array(b_values)
