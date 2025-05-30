r"""Isoradial lines.

This module provides the :py:class:`Isoradial` class, which is used to calculate and visualize isoradial lines.
An isoradial line is a line of constant radius in the black hole frame :math:`(r, \alpha)`.
This can be used to calculate how such line appears in the observer's plane :math:`(b, \alpha)`, essentailly
visualizing the spacetime curvature.
"""

from typing import Dict

from matplotlib.axes import Axes

import numpy as np

from luminet import black_hole_math as bhmath
from luminet.solver import improve_solutions, interpolator
from luminet.viz import colorline
from luminet.transform import polar_to_cartesian


class Isoradial:
    """Calculate and visualize isoradial lines.
    """
    def __init__(
        self,
        radius: float,
        incl: float,
        bh_mass: float,
        order: int = 0,
        acc: float | None = None,
        params: Dict | None = None,
    ):
        r"""
        Args:
            radius (float): Radius of the isoradial in the black hole frame :math:`(r, \alpha)`
            incl (float): Inclination angle of the observer with respect to the black hole.
                :math:`0` degrees corresponds to the observer looking top-down on the black hole.
                :math:`\pi/2` corresponds to the observer looking at the black hole edge-on.
            bh_mass (float): Mass of the black hole
            order (int): Order of the isoradial. 
                :math:`0` corresponds to direct images, 
                :math:`1` to the first-order image i.e. "ghost" image.
                Default is :math:`0`.
            acc (float): Accretion rate of the black hole. Default is None.
            params (Dict): Additional parameters for the isoradial calculation.
                Default is None.
        """
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

        self.radii_b = []
        """np.ndarray: Radii of the isoradial in the observer plane :math:`b`."""
        self.angles = []
        r"""np.ndarray: Angles of the isoradial in black hole / observer frame :math:`\alpha`."""
        self.redshift_factors = None
        """np.ndarray: Redshift factors of the isoradial :math:`(1 + z)`."""

        self.calculate()

    def calculate_coordinates(self):
        """Calculates the angles :math:`alpha` and radii :math:`b` of the isoradial.

        Saves these values in the :py:attr:`angles` and :py:attr:`radii_b` attributes.

        Returns:
            Tuple[np.ndarray]: 
                Tuple containing the angles and radiifor the image on the observer plane.
        """

        angles = []
        impact_parameters = []
        t = np.linspace(0, 2 * np.pi, self.params["angular_precision"])
        for alpha in t:
            b = bhmath.solve_for_impact_parameter(
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
        """Calculates the redshift factor :math:`(1 + z)` over the isoradial

        Saves these values in the :py:attr:`redshift_factors` attribute.
        
        Returns:
            np.ndarray: Redshift factors of the isoradial
        """
        redshift_factors = bhmath.calc_redshift_factor(
            radius=self.radius,
            angle=self.angles,
            incl=self.incl,
            bh_mass=self.bh_mass,
            b=self.radii_b,
        )
        self.redshift_factors = np.array(redshift_factors)
        return redshift_factors

    def calculate(self):
        """Calculates the coordinates and redshift factors on the isoradial line.
        
        See also:
            :py:meth:`calculate_coordinates` and
            :py:meth:`calc_redshift_factors`

        Returns:
            :py:class:`Isoradial`: The :py:class:`Isoradial` object itself, but with calculated coordinates and redshift factors.
        """
        self.calculate_coordinates()
        self.calc_redshift_factors()
        return self

    def get_b_from_angle(self, angle: float | np.ndarray):
        r"""Get the impact parameter :math:`b` for a given angle :math:`\alpha` on the isoradial.

        This method does not calculate the impact parameter, but rather finds the closest
        impact parameter to the given angle.

        Args:
            angle (float | np.ndarray): Angle :math:`\alpha` in radians.

        See also:
            :py:meth:`calc_b_from_angle` to explicitly solve for the impact parameter.

        Returns:
            float | np.ndarray: The impact parameter :math:`b` for the given angle :math:`\alpha`.

        """
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

    def solve_for_b_from_angle(self, angle: float):
        r"""Calculate the impact parameter :math:`b` for a given angle :math:`\alpha` on the isoradial.
        
        This method solves for the impact parameter :math:`b` for a given angle :math:`\alpha` on the isoradial.

        Args:
            angle (float): Angle :math:`\alpha` in radians.

        Returns:
            float: The impact parameter :math:`b` for the given angle :math:`\alpha`.
        """
        b = bhmath.solve_for_impact_parameter(
            radius=self.radius,
            incl=self.incl,
            alpha=angle,
            bh_mass=self.bh_mass,
            order=self.order,
        )
        return b

    def plot(self, ax: Axes, z, cmap, norm, **kwargs):
        """Plot the isoradial.

        Args:
            ax (:py:class:`~matplotlib.axes.Axes`): The axis on which the isoradial should be plotted
            z (array-like): The color values to be used.
            cmap (str): The colormap to be used. Must be a valid matplotlib colormap string.
            norm (tuple): The normalization to be used.
            **kwargs: Additional arguments to be passed to the colorline function

        Returns:
            :py:class:`~matplotlib.axes.Axes`: The axis with the isoradial plotted.
        """
        ax = colorline(
            ax, self.angles, self.radii_b, z=z, cmap=cmap, norm=norm, **kwargs
        )

        return ax

    def _has_redshift(self, z):
        """Calculate if the class theoretically contains redshift value :math:`z`
        
        :meta private:
        """
        # TODO: math

    def calc_redshift_locations(self, redshift):
        """Calculates which location on the isoradial has some redshift value (not redshift factor)

        Args:
            redshift (float): Redshift value

        Returns:
            Tuple[np.ndarray]: 2-tuple containing the angles and radii for the redshift value.
                In general, either two or zero solutions exist.
        """
        # Find all zero crossings
        angles = np.linspace(0, 2 * np.pi, 100)
        impact_parameters = [self.solve_for_b_from_angle(angle) for angle in angles]
        b_interp = interpolator(angles, impact_parameters)
        
        # function to solve for
        func = (
            lambda angle: redshift
            + 1
            - bhmath.calc_redshift_factor(
                self.radius,
                angle,
                self.incl,
                self.bh_mass,
                b_interp(angle),
            )
        )

        values = [func(angle) for angle in angles]
        sign_changes = np.where(np.diff(np.sign(values)))[0]

        solutions = [None, None]  # Initialize with None to ensure two solutions
        for idx in sign_changes:
            # Find the root of the function in the interval
            angle0, angle1 = angles[idx], angles[idx + 1]
            z0, z1 = values[idx], values[idx + 1]
            try:
                angle = improve_solutions(func, (angle0, angle1), (z0, z1), kwargs={})
                # split solutions based on their angle: useful for plotting later on
                y_value = polar_to_cartesian(angle, b_interp(angle))[1]
                if z0 < 0:
                    solutions[0] = angle
                elif z1 < 0:
                    solutions[1] = angle
            except Exception as e:
                # If brentq fails to find a root in the interval, skip it
                pass

        # Calculate corresponding b values for the found angles
        b_values = [
            self.solve_for_b_from_angle(angle) if angle is not None else None
            for angle in solutions
        ]

        assert (
            len(b_values) == len(solutions) == 2
        ), "Should have found 2 solutions, or at least padded the second solution with None"
        return np.array(solutions), np.array(b_values)
